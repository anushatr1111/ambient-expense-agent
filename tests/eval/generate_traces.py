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

import json
import os
import sys
import uuid
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

# Add the project root to python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Mock the Gemini Client for review_risk_node to make trace generation fast, reliable and offline-safe
class MockClient:
    def __init__(self, *args, **kwargs):
        self.aio = MagicMock()
        self.aio.models.generate_content = AsyncMock(side_effect=self.generate_content)

    async def generate_content(self, model, contents, config, **kwargs):
        mock_response = MagicMock()
        has_risks = False
        concerns = []
        explanation = "The expense report was assessed and appears to be normal."
        
        try:
            if "luxury" in contents.lower():
                has_risks = True
                concerns.append("High risk category")
                explanation = "Luxury category expenses require additional corporate scrutiny."
        except Exception:
            pass

        mock_response.text = json.dumps({
            "has_risks": has_risks,
            "concerns": concerns,
            "explanation": explanation
        })
        return mock_response

# Apply the Client mock patch
patcher = patch("expense_agent.agent.Client", MockClient)
patcher.start()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from expense_agent.agent import root_agent

async def run_eval_case(case):
    session_service = InMemorySessionService()
    session = await session_service.create_session(user_id="eval_user", app_name="expense_agent")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="expense_agent")

    prompt_text = case["prompt"]["parts"][0]["text"]
    expense_data = json.loads(prompt_text)

    # Initial user turn
    turns = []
    turns.append({
        "turnIndex": len(turns),
        "turnId": str(uuid.uuid4()),
        "events": [{
            "author": "user",
            "content": {
                "role": "user",
                "parts": [{"text": prompt_text}]
            }
        }]
    })

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt_text)]
    )

    # Run the workflow first time
    events = []
    paused_on_hilt = False
    async for event in runner.run_async(
        new_message=message,
        user_id="eval_user",
        session_id=session.id
    ):
        events.append(event)
        
        # Collect model responses
        if event.content and event.content.parts:
            text_parts = [part.text for part in event.content.parts if hasattr(part, "text") and part.text]
            if text_parts:
                turns.append({
                    "turnIndex": len(turns),
                    "turnId": str(uuid.uuid4()),
                    "events": [{
                        "author": "model",
                        "content": {
                            "role": "model",
                            "parts": [{"text": t} for t in text_parts]
                        }
                    }]
                })

        # Collect function calls representing human approval pause
        for fc in event.get_function_calls():
            if fc.name == "adk_request_input" and fc.id == "human_decision":
                paused_on_hilt = True
                turns.append({
                    "turnIndex": len(turns),
                    "turnId": str(uuid.uuid4()),
                    "events": [{
                        "author": "model",
                        "content": {
                            "role": "model",
                            "parts": [{
                                "functionCall": {
                                    "name": fc.name,
                                    "args": fc.args,
                                    "id": fc.id
                                }
                            }]
                        }
                    }]
                })

    final_output = None
    if paused_on_hilt:
        # Automate decision:
        # - Reject prompt injections
        # - Approve clean requests
        description = expense_data.get("description", "").lower()
        injection_keywords = ["ignore", "override", "bypass", "auto-approve", "autoapprove"]
        is_injection = any(kw in description for kw in injection_keywords)
        
        decision = "reject" if is_injection else "approve"
        
        # Collect user's resume response turn
        turns.append({
            "turnIndex": len(turns),
            "turnId": str(uuid.uuid4()),
            "events": [{
                "author": "user",
                "content": {
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": "human_decision",
                            "response": {"human_decision": decision},
                            "id": "human_decision"
                        }
                    }]
                }
            }]
        })

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

        # Resume the session
        async for event in runner.run_async(
            new_message=resume_message,
            user_id="eval_user",
            session_id=session.id
        ):
            events.append(event)
            if event.content and event.content.parts:
                text_parts = [part.text for part in event.content.parts if hasattr(part, "text") and part.text]
                if text_parts:
                    turns.append({
                        "turnIndex": len(turns),
                        "turnId": str(uuid.uuid4()),
                        "events": [{
                            "author": "model",
                            "content": {
                                "role": "model",
                                "parts": [{"text": t} for t in text_parts]
                            }
                        }]
                    })
            if event.output is not None:
                final_output = event.output
    else:
        # Non-HILT execution paths
        for event in events:
            if event.output is not None:
                final_output = event.output

    response_text = json.dumps(final_output) if final_output else ""

    # Return valid evaluation case format matching Pydantic alias expectations
    return {
        "evalCaseId": case["eval_case_id"],
        "prompt": case["prompt"],
        "responses": [
            {
                "response": {
                    "role": "model",
                    "parts": [{"text": response_text}]
                }
            }
        ],
        "agentData": {
            "turns": turns
        }
    }

async def main():
    dataset_path = "tests/eval/datasets/basic-dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    traces = []
    for case in dataset["eval_cases"]:
        print(f"Generating trace for case: {case['eval_case_id']}...")
        trace = await run_eval_case(case)
        traces.append(trace)

    output_dir = "artifacts/traces"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_traces.json")
    
    # Save as EvaluationDataset format
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"evalCases": traces}, f, indent=2)

    print(f"Trace generation completed! Traces saved at: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
