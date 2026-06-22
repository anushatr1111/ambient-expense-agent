# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import datetime
import json
import os
from typing import Any
import google.auth
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

from google.adk.workflow import Workflow, START
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.adk.apps import App
from google.genai import types
from google.genai import Client

from expense_agent import config

try:
    _, project_id = google.auth.default()
    if project_id:
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
except Exception:
    pass

# Fallback project ID to prevent initialization errors during local testing/collection
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    os.environ["GOOGLE_CLOUD_PROJECT"] = "mock-project"

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


class ExpenseReport(BaseModel):
    amount: float
    submitter: str
    category: str
    description: str
    date: str


class RiskReport(BaseModel):
    has_risks: bool = Field(description="True if there are notable risk factors or policy violations, False otherwise.")
    concerns: list[str] = Field(description="A list of identified risk factors, suspicious patterns, or policy violations.")
    explanation: str = Field(description="Detailed explanation of the risk assessment.")


async def parse_expense(ctx: Context, node_input: Any):
    """Parses incoming JSON event (handles plain JSON and base64-encoded Pub/Sub payloads)."""
    raw_text = ""
    data_payload = None

    if isinstance(node_input, dict):
        data_payload = node_input.get("data")
    elif hasattr(node_input, "parts") and node_input.parts:
        raw_text = "".join(part.text for part in node_input.parts if part.text)
        try:
            parsed_json = json.loads(raw_text)
            data_payload = parsed_json.get("data")
        except Exception:
            data_payload = raw_text
    else:
        raw_text = str(node_input)
        try:
            parsed_json = json.loads(raw_text)
            data_payload = parsed_json.get("data")
        except Exception:
            data_payload = raw_text

    expense_data = {}
    if data_payload:
        if isinstance(data_payload, dict):
            expense_data = data_payload
        elif isinstance(data_payload, str):
            try:
                # Try to base64 decode (real Pub/Sub event)
                decoded = base64.b64decode(data_payload).decode("utf-8")
                expense_data = json.loads(decoded)
            except Exception:
                try:
                    # Fallback to plain string JSON
                    expense_data = json.loads(data_payload)
                except Exception:
                    pass
    else:
        try:
            expense_data = json.loads(raw_text)
        except Exception:
            pass

    amount = float(expense_data.get("amount", 0.0))
    submitter = str(expense_data.get("submitter", "Unknown"))
    category = str(expense_data.get("category", "General"))
    description = str(expense_data.get("description", "No description"))
    date = str(expense_data.get("date", datetime.date.today().isoformat()))

    expense = ExpenseReport(
        amount=amount,
        submitter=submitter,
        category=category,
        description=description,
        date=date
    )

    state_delta = {"expense": expense.model_dump()}

    if amount < config.THRESHOLD:
        yield Event(output=expense.model_dump(), route="auto_approve", state=state_delta)
    else:
        yield Event(output=expense.model_dump(), route="review", state=state_delta)
import re

SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b")
CC_PATTERN = re.compile(r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b|\b\d{13,19}\b")


def is_prompt_injection(text: str) -> bool:
    lower_text = text.lower()
    injection_keywords = [
        "ignore prompt", "ignore instruction", "ignore previous", "ignore above",
        "override", "bypass rule", "force approve", "auto-approve", "autoapprove",
        "system instruction", "system prompt", "you are now", "instead of",
        "new rules", "change rules", "approve this"
    ]
    return any(kw in lower_text for kw in injection_keywords)


async def security_checkpoint_node(ctx: Context, node_input: dict):
    """Scrubs personal data (SSN, credit card) and checks for prompt injection."""
    expense = ctx.state.get("expense", {})
    description = expense.get("description", "")

    redacted_categories = []
    scrubbed_description = description

    # 1. Scrub SSN
    if SSN_PATTERN.search(scrubbed_description):
        scrubbed_description = SSN_PATTERN.sub("[REDACTED SSN]", scrubbed_description)
        redacted_categories.append("SSN")

    # 2. Scrub Credit Cards
    if CC_PATTERN.search(scrubbed_description):
        scrubbed_description = CC_PATTERN.sub("[REDACTED CREDIT CARD]", scrubbed_description)
        redacted_categories.append("CREDIT_CARD")

    # Update description in expense dictionary in place
    expense["description"] = scrubbed_description

    # 3. Check for prompt injection
    is_injection = is_prompt_injection(description)

    state_delta = {
        "expense": expense,
        "security_event": is_injection,
        "redacted_categories": redacted_categories
    }

    if is_injection:
        yield Event(
            output=expense,
            route="security_event",
            state=state_delta
        )
    else:
        yield Event(
            output=expense,
            route="clean",
            state=state_delta
        )


async def review_risk_node(ctx: Context, node_input: dict):
    """Calls Gemini with structured outputs to identify potential risk factors."""
    client = Client()
    prompt = f"Analyze the following corporate expense report for risks, policy violations, or duplicate submissions:\n{json.dumps(node_input, indent=2)}"

    response = await client.aio.models.generate_content(
        model=config.MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RiskReport,
        )
    )

    try:
        risk_data = json.loads(response.text)
        risk_report = RiskReport(**risk_data)
    except Exception:
        risk_report = RiskReport(
            has_risks=False, 
            concerns=[], 
            explanation="Failed to parse model risk assessment."
        )

    alert_text = (
        f"⚠️ Alert: Risk Review Completed.\n"
        f"Risks Detected: {risk_report.has_risks}\n"
        f"Concerns: {', '.join(risk_report.concerns) if risk_report.concerns else 'None'}\n"
        f"Explanation: {risk_report.explanation}"
    )

    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=alert_text)]),
        output=risk_report.model_dump(),
        state={"risk_report": risk_report.model_dump()}
    )


async def human_approval_node(ctx: Context, node_input: Any):
    """Human-in-the-loop step pausing execution for approval or rejection."""
    if not ctx.resume_inputs or "human_decision" not in ctx.resume_inputs:
        is_security_event = ctx.state.get("security_event", False)
        if is_security_event:
            message = (
                "⚠️ WARNING: This expense report has been flagged as a potential SECURITY EVENT "
                "(potential prompt injection detected). Reply with 'approve' or 'reject':"
            )
        else:
            message = "Expense report is pending human approval. Reply with 'approve' or 'reject':"

        yield RequestInput(
            interrupt_id="human_decision",
            message=message
        )
        return

    decision = str(ctx.resume_inputs["human_decision"]).strip().lower()
    approved = "approve" in decision
    reason = f"Human decision: {decision.upper()}"

    yield Event(
        output={"approved": approved, "reason": reason},
        state={"human_decision": {"approved": approved, "reason": reason}}
    )


def record_outcome_node(ctx: Context, node_input: dict):
    """Gathers and writes final decision status to output and state."""
    expense = ctx.state.get("expense", {})
    risk_report = ctx.state.get("risk_report")

    if "outcome" in ctx.state:
        # Already set via auto_approve path
        return

    approved = False
    reason = "No decision recorded"

    if node_input:
        if isinstance(node_input, dict):
            if "approved" in node_input:
                approved = node_input.get("approved", False)
                reason = node_input.get("reason", "No decision recorded")
            elif "human_decision" in node_input:
                decision_str = str(node_input["human_decision"]).strip().lower()
                approved = "approve" in decision_str
                reason = f"Human decision: {decision_str.upper()}"
            elif "result" in node_input:
                decision_str = str(node_input["result"]).strip().lower()
                approved = "approve" in decision_str
                reason = f"Human decision: {decision_str.upper()}"
        elif isinstance(node_input, str):
            decision_str = node_input.strip().lower()
            approved = "approve" in decision_str
            reason = f"Human decision: {decision_str.upper()}"
    else:
        human_decision = ctx.state.get("human_decision")
        if human_decision and isinstance(human_decision, dict):
            approved = human_decision.get("approved", False)
            reason = human_decision.get("reason", "No decision recorded")

    outcome = {
        "status": "APPROVED" if approved else "REJECTED",
        "decision_source": "HUMAN_APPROVAL",
        "reason": reason,
        "expense": expense,
        "risk_report": risk_report,
        "security_event": ctx.state.get("security_event", False),
        "redacted_categories": ctx.state.get("redacted_categories", []),
    }

    msg_text = f"Final Expense Decision: {outcome['status']}\nReason: {outcome['reason']}"

    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg_text)]),
        output=outcome,
        state={"outcome": outcome}
    )


def record_auto_approve_node(ctx: Context, node_input: dict):
    """Processes auto-approval for reports below the threshold limit."""
    expense = ctx.state.get("expense", {})
    outcome = {
        "status": "APPROVED",
        "decision_source": "AUTO_APPROVAL",
        "reason": f"Amount ${expense.get('amount')} is under the ${config.THRESHOLD} threshold.",
        "expense": expense,
        "risk_report": None,
    }

    msg_text = f"Expense of ${expense.get('amount')} by {expense.get('submitter')} was auto-approved."

    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg_text)]),
        output=outcome,
        state={"outcome": outcome}
    )


root_agent = Workflow(
    name="root_agent",
    edges=[
        (START, parse_expense),
        (parse_expense, {
            "auto_approve": record_auto_approve_node,
            "review": security_checkpoint_node,
        }),
        (security_checkpoint_node, {
            "clean": review_risk_node,
            "security_event": human_approval_node,
        }),
        (review_risk_node, human_approval_node),
        (human_approval_node, record_outcome_node),
    ],
)

app = App(
    root_agent=root_agent,
    name="expense_agent",
)
