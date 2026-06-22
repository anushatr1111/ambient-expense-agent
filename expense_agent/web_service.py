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
import json
import logging
import os
import uuid
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Configure environment variables to respect developer checklist
os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get("GOOGLE_CLOUD_PROJECT", "mock-project")
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["OTEL_TO_CLOUD"] = "False"

from expense_agent.agent import root_agent

# Use standard Python logging for console logs as requested
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ambient_expense_agent")

app = FastAPI(title="Ambient Expense Agent Web Service")

# Initialize ADK Session Service and Runner
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, session_service=session_service, app_name="expense_agent")

class PubSubMessage(BaseModel):
    data: Optional[str] = None
    attributes: Optional[dict[str, str]] = None
    messageId: Optional[str] = None
    publishTime: Optional[str] = None

class PubSubPayload(BaseModel):
    message: PubSubMessage
    subscription: Optional[str] = None

class ResumeRequest(BaseModel):
    decision: str = Field(description="Decision from human: 'approve' or 'reject'")

@app.post("/apps/expense_agent/trigger/pubsub")
@app.post("/")
async def trigger_pubsub(payload: PubSubPayload):
    # Gotcha: Pub/Sub sends fully-qualified subscription path
    # e.g., projects/project-id/subscriptions/sub-id
    # Normalize to short name to keep session records readable
    subscription_path = payload.subscription or "projects/mock-project/subscriptions/expense-sub"
    short_sub_name = subscription_path.split("/")[-1]
    
    logger.info(f"Received Pub/Sub message. subscription={subscription_path} (normalized to: {short_sub_name})")

    # Decode base64 data payload
    decoded_data = ""
    if payload.message.data:
        try:
            # Try to decode base64
            decoded_bytes = base64.b64decode(payload.message.data)
            decoded_data = decoded_bytes.decode("utf-8")
        except Exception:
            # Fallback to direct string if not base64
            decoded_data = payload.message.data
    else:
        logger.error("Pub/Sub message contains no data")
        raise HTTPException(status_code=400, detail="Missing message data")

    # The agent runs using the decoded event data as a user message
    new_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=decoded_data)]
    )

    # Use the normalized short subscription name as user_id to keep records clean
    user_id = short_sub_name
    session_id = str(uuid.uuid4())

    logger.info(f"Starting workflow run for user_id={user_id}, session_id={session_id}")
    
    # Create the session
    session = await session_service.create_session(
        app_name="expense_agent",
        user_id=user_id,
        session_id=session_id
    )

    events = []
    paused_on_hilt = False
    hilt_message = ""

    try:
        async for event in runner.run_async(
            new_message=new_message,
            user_id=user_id,
            session_id=session.id
        ):
            events.append(event)
            # Check if we hit a RequestInput interrupt (HILT)
            for fc in event.get_function_calls():
                if fc.name == "adk_request_input" and fc.id == "human_decision":
                    paused_on_hilt = True
                    hilt_message = fc.args.get("message") if fc.args else "Pending human approval"
    except Exception as e:
        logger.exception(f"Error running workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {e}")

    # Extract final output if completed
    final_output = None
    if not paused_on_hilt:
        for event in events:
            if event.output is not None:
                final_output = event.output

    response_payload = {
        "status": "PAUSED_FOR_APPROVAL" if paused_on_hilt else "COMPLETED",
        "session_id": session.id,
        "user_id": user_id,
    }
    if paused_on_hilt:
        response_payload["message"] = hilt_message
        response_payload["resume_url"] = f"/apps/expense_agent/sessions/{session.id}/resume"
        logger.info(f"Workflow paused at HITL approval. Session ID: {session.id}. Resume via POST to {response_payload['resume_url']}")
    else:
        response_payload["outcome"] = final_output
        logger.info(f"Workflow completed instantly. Outcome: {final_output}")

    return response_payload

@app.post("/apps/expense_agent/sessions/{session_id}/resume")
async def resume_session(session_id: str, req: ResumeRequest):
    # Resume the HILT step with the decision
    decision = req.decision.strip().lower()
    if decision not in ["approve", "reject"]:
        raise HTTPException(status_code=400, detail="Decision must be 'approve' or 'reject'")

    # Retrieve the session
    session = None
    response = await session_service.list_sessions(app_name="expense_agent")
    for s in response.sessions:
        if s.id == session_id:
            session = s
            break
            
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    logger.info(f"Resuming session {session_id} with decision: {decision}")

    # Create the resume response message
    resume_message = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name="human_decision",
                    response={"human_decision": decision},
                    id="human_decision",
                )
            )
        ]
    )

    events = []
    try:
        async for event in runner.run_async(
            new_message=resume_message,
            user_id=session.user_id,
            session_id=session.id
        ):
            events.append(event)
    except Exception as e:
        logger.exception(f"Error resuming workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Resuming workflow failed: {e}")

    final_output = None
    for event in events:
        if event.output is not None:
            final_output = event.output

    return {
        "status": "COMPLETED",
        "session_id": session.id,
        "outcome": final_output
    }

@app.get("/apps/expense_agent/sessions")
async def list_active_sessions():
    response = await session_service.list_sessions(app_name="expense_agent")
    result = []
    for s in response.sessions:
        result.append({
            "session_id": s.id,
            "user_id": s.user_id,
            "created_at": s.created_at.isoformat() if hasattr(s.created_at, "isoformat") else str(s.created_at)
        })
    return result

