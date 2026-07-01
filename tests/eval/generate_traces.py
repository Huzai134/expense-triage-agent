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

import asyncio
import json
import os

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent


def normalize_author(author: str) -> str:
    """Normalize event author to keep evaluation traces readable."""
    if author == "user":
        return "user"
    if author == "tool":
        return "tool"
    if "expense_triage_workflow" in author:
        return "expense_triage_workflow"
    return author


def serialize_content(content: types.Content | None) -> dict | None:
    """Convert ADK content structure to standard evaluation JSON schema."""
    if not content or not content.parts:
        return None
    parts_list = []
    for p in content.parts:
        part_dict = {}
        if p.text:
            part_dict["text"] = p.text
        elif p.function_call:
            part_dict["function_call"] = {
                "name": p.function_call.name,
                "args": p.function_call.args,
            }
        elif p.function_response:
            part_dict["function_response"] = {
                "name": p.function_response.name,
                "response": p.function_response.response,
            }
        parts_list.append(part_dict)
    return {"role": content.role, "parts": parts_list}


async def generate_trace_for_case(case: dict) -> dict:
    """Run a scenario through the agent, automating human-in-the-loop and collecting events."""
    eval_case_id = case["eval_case_id"]
    prompt_text = case["prompt"]["parts"][0]["text"]

    session_service = InMemorySessionService()
    session = await session_service.create_session(user_id="test_user", app_name="app")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="app")

    initial_message = types.Content(
        role="user", parts=[types.Part.from_text(text=prompt_text)]
    )

    turns = []
    current_turn_events = []
    turn_index = 0

    # Turn 0: User input
    current_turn_events.append(
        {
            "author": "user",
            "content": {"role": "user", "parts": [{"text": prompt_text}]},
        }
    )

    suspended = False
    interrupt_id = "human_decision"

    # Execute first turn
    async for event in runner.run_async(
        new_message=initial_message,
        user_id="test_user",
        session_id=session.id,
    ):
        if event.content:
            current_turn_events.append(
                {
                    "author": normalize_author(event.author),
                    "content": serialize_content(event.content),
                }
            )

            # Check if this node is requesting input (HITL)
            if event.content.parts:
                for part in event.content.parts:
                    if (
                        part.function_call
                        and part.function_call.name == "adk_request_input"
                    ):
                        suspended = True
                        interrupt_id = (
                            part.function_call.args.get("interruptId")
                            or "human_decision"
                        )

    # Save turn 0
    turns.append({"turn_index": turn_index, "events": current_turn_events})

    # Resumption phase (Turn 1) if HITL was triggered
    if suspended:
        turn_index += 1
        decision = "reject" if "reject" in eval_case_id else "approve"

        resume_message = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="adk_request_input",
                        id=interrupt_id,
                        response={"response": decision},
                    )
                )
            ],
        )

        resume_events = []
        resume_events.append(
            {
                "author": "user",
                "content": serialize_content(resume_message),
            }
        )

        # Resume the workflow run by passing the resumption message as new_message
        async for event in runner.run_async(
            new_message=resume_message,
            user_id="test_user",
            session_id=session.id,
        ):
            if event.content:
                resume_events.append(
                    {
                        "author": normalize_author(event.author),
                        "content": serialize_content(event.content),
                    }
                )

        turns.append({"turn_index": turn_index, "events": resume_events})

    return {
        "eval_case_id": eval_case_id,
        "prompt": case["prompt"],
        "agent_data": {
            "agents": {
                "expense_triage_workflow": {
                    "agent_id": "expense_triage_workflow",
                    "instruction": "Triage and review expenses",
                }
            },
            "turns": turns,
        },
    }


async def main() -> None:
    dataset_path = "tests/eval/datasets/basic-dataset.json"
    with open(dataset_path) as f:
        dataset = json.load(f)

    trace_cases = []
    for case in dataset["eval_cases"]:
        print(f"Running case: {case['eval_case_id']}...")
        trace_case = await generate_trace_for_case(case)
        trace_cases.append(trace_case)

    output_dir = "artifacts/traces"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_traces.json")

    with open(output_path, "w") as f:
        json.dump({"eval_cases": trace_cases}, f, indent=2)
    print(f"Traces written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
