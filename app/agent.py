# ruff: noqa
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

import os
import re
import base64
import json
from dotenv import load_dotenv

# Load env variables from .env if present
load_dotenv()

# Determine if we should use Vertex AI
use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() in ("true", "1")

if use_vertex:
    import google.auth

    try:
        _, project_id = google.auth.default()
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
    except Exception:
        # Fallback if default project detection fails
        pass
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
else:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

from google.adk.workflow import Workflow, START
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.genai import types
from pydantic import BaseModel, Field
from typing import Any

# ------------------------------------------------------------------------------
# 1. Pydantic Schemas
# ------------------------------------------------------------------------------


class EventInput(BaseModel):
    data: Any = Field(
        description="The JSON event details, which can be base64-encoded string or plain dictionary."
    )


class Expense(BaseModel):
    amount: float = Field(description="The expense amount in USD.")
    submitter: str = Field(description="The name of the submitter.")
    category: str = Field(description="The category of the expense.")
    description: str = Field(description="The description/purpose of the expense.")
    date: str = Field(description="The date of the transaction.")


class ReviewResult(BaseModel):
    is_compliant: bool = Field(
        description="True if the expense complies with company spending policies, False otherwise."
    )
    risk_analysis: str = Field(
        description="A brief explanation of compliance status and any flagged risks."
    )


# ------------------------------------------------------------------------------
# 2. Node Implementations (Functions and Agents)
# ------------------------------------------------------------------------------


def parse_expense_event(node_input: EventInput) -> Event:
    """Parses the input event data (decoding base64 if needed) into an Expense object."""
    raw_data = node_input.data

    if isinstance(raw_data, str):
        # Try base64 decoding first
        try:
            decoded = base64.b64decode(raw_data).decode("utf-8")
            parsed = json.loads(decoded)
        except Exception:
            # Fallback to plain JSON string
            try:
                parsed = json.loads(raw_data)
            except Exception as e:
                raise ValueError(f"Could not parse data string: {e}")
    elif isinstance(raw_data, dict):
        parsed = raw_data
    else:
        raise ValueError(f"Unsupported data type for event data: {type(raw_data)}")

    expense = Expense(
        amount=float(parsed["amount"]),
        submitter=parsed["submitter"],
        category=parsed["category"],
        description=parsed["description"],
        date=parsed["date"],
    )

    state_updates = {
        "amount": expense.amount,
        "submitter": expense.submitter,
        "category": expense.category,
        "description": expense.description,
        "date": expense.date,
        "redacted_categories": [],
    }

    return Event(output=expense, actions=EventActions(state_delta=state_updates))


def route_expense(node_input: Expense) -> Event:
    """Routes the expense based on transaction value."""
    if node_input.amount < 100.0:
        return Event(output=node_input, actions=EventActions(route="low_value"))
    else:
        return Event(output=node_input, actions=EventActions(route="high_value"))


def auto_approve(node_input: Expense) -> Event:
    """Deterministic auto-approval for low-value expenses (< $100)."""
    message_text = f"Low-value expense of ${node_input.amount:.2f} submitted by {node_input.submitter} for {node_input.category} is auto-approved."
    return Event(
        output={
            "status": "approved",
            "reason": "Low-value expense auto-approved.",
            "amount": node_input.amount,
        },
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=message_text)]
        ),
    )


def security_screen(node_input: Expense) -> Event:
    """Pre-LLM security check to redact PII and prevent prompt injection."""
    desc = node_input.description
    submitter = node_input.submitter

    redacted_categories = set()

    # Redact PII: email addresses
    email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
    if re.search(email_pattern, desc) or re.search(email_pattern, submitter):
        redacted_categories.add("Email")
    desc = re.sub(email_pattern, "[REDACTED EMAIL]", desc)
    submitter = re.sub(email_pattern, "[REDACTED EMAIL]", submitter)

    # Redact PII: 10-digit phone numbers
    phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
    if re.search(phone_pattern, desc) or re.search(phone_pattern, submitter):
        redacted_categories.add("Phone")
    desc = re.sub(phone_pattern, "[REDACTED PHONE]", desc)
    submitter = re.sub(phone_pattern, "[REDACTED PHONE]", submitter)

    # Redact PII: SSNs
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    if re.search(ssn_pattern, desc) or re.search(ssn_pattern, submitter):
        redacted_categories.add("SSN")
    desc = re.sub(ssn_pattern, "[REDACTED SSN]", desc)
    submitter = re.sub(ssn_pattern, "[REDACTED SSN]", submitter)

    # Redact PII: Credit Card numbers
    ccn_pattern = r"\b(?:\d[ -]*?){13,16}\b"
    if re.search(ccn_pattern, desc) or re.search(ccn_pattern, submitter):
        redacted_categories.add("Credit Card")
    desc = re.sub(ccn_pattern, "[REDACTED CREDIT CARD]", desc)
    submitter = re.sub(ccn_pattern, "[REDACTED CREDIT CARD]", submitter)

    # Redact PII: Names (tokens of the submitter name)
    name_parts = re.split(r"\s+", node_input.submitter.strip())
    redacted_name_in_desc = False
    for part in name_parts:
        if len(part) > 1 and part.lower() not in {
            "and",
            "the",
            "for",
            "mr",
            "ms",
            "mrs",
            "dr",
            "inc",
            "corp",
            "ltd",
        }:
            name_pattern = rf"\b{re.escape(part)}\b"
            if re.search(name_pattern, desc, re.IGNORECASE):
                redacted_name_in_desc = True
            desc = re.sub(name_pattern, "[REDACTED NAME]", desc, flags=re.IGNORECASE)

    if redacted_name_in_desc or node_input.submitter.strip() != "":
        redacted_categories.add("Name")

    redacted_submitter = "[REDACTED NAME]"

    injection_patterns = [
        r"ignore\s+(?:any|previous|the)?\s*instruction",
        r"system\s+prompt",
        r"you\s+must\s+now",
        r"bypass\s+(?:all\s+)?(?:policy|rules)",
        r"override\s+(?:all\s+)?(?:policy|rules)",
    ]
    is_injection = any(
        re.search(pattern, desc, re.IGNORECASE) for pattern in injection_patterns
    )

    # Save cleaned parameters into state for LLM reference and human review
    state_updates = {
        "amount": node_input.amount,
        "submitter": redacted_submitter,
        "category": node_input.category,
        "description": desc,
        "date": node_input.date,
        "redacted_categories": list(redacted_categories),
    }

    if is_injection:
        state_updates["security_event"] = True
        security_event_result = ReviewResult(
            is_compliant=False,
            risk_analysis="SECURITY ALERT: Potential prompt injection detected. Compliance analysis bypassed for safety.",
        )
        return Event(
            output=security_event_result,
            actions=EventActions(route="security_flagged", state_delta=state_updates),
        )

    cleaned_expense = Expense(
        amount=node_input.amount,
        submitter=redacted_submitter,
        category=node_input.category,
        description=desc,
        date=node_input.date,
    )
    return Event(
        output=cleaned_expense,
        actions=EventActions(route="security_passed", state_delta=state_updates),
    )


def handle_rejection(node_input: dict) -> Event:
    """Terminal node for security policy violations."""
    return Event(output=node_input)


review_agent = LlmAgent(
    name="review_agent",
    model=Gemini(model="gemini-2.5-flash"),
    instruction="""You are a compliance officer. Analyze the following expense for policy violations and risk (such as personal spending, luxury travel, or suspicious transactions).
    Expense Details:
    Submitter: {submitter}
    Category: {category}
    Amount: ${amount}
    Date: {date}
    Description: {description}
    """,
    output_schema=ReviewResult,
    output_key="review_report",
)


async def human_review_pause(ctx: Context, node_input: ReviewResult):
    """HITL step using RequestInput to pause and await reviewer authorization."""
    if not ctx.resume_inputs or "human_decision" not in ctx.resume_inputs:
        is_security = ctx.state.get("security_event", False)
        redacted = ctx.state.get("redacted_categories", [])
        redacted_str = f" (Redacted: {', '.join(redacted)})" if redacted else ""

        if is_security:
            message = (
                f"SECURITY WARNING: Prompt injection was detected in this expense! "
                f"LLM review was bypassed. Risk Analysis: {node_input.risk_analysis}{redacted_str}. "
                f"Do you approve this expense of ${ctx.state.get('amount')}? (reply 'approve' or 'reject')"
            )
        else:
            message = (
                f"Compliance check completed. Risk Analysis: {node_input.risk_analysis}{redacted_str}. "
                f"Do you approve this expense of ${ctx.state.get('amount')}? (reply 'approve' or 'reject')"
            )

        yield RequestInput(interrupt_id="human_decision", message=message)
        return

    decision = ctx.resume_inputs["human_decision"].lower().strip()
    is_security = ctx.state.get("security_event", False)

    if "approve" in decision:
        reason = "Approved by human reviewer."
        if is_security:
            reason += " WARNING: Submitter attempted prompt injection."
        reason += f" Risk Analysis: {node_input.risk_analysis}"

        yield Event(
            output={
                "status": "approved",
                "reason": reason,
                "amount": ctx.state.get("amount"),
            },
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=f"Expense of ${ctx.state.get('amount')} was approved by human reviewer. {reason}"
                    )
                ],
            ),
        )
    else:
        reason = "Rejected by human reviewer."
        if is_security:
            reason += " Security alert: Prompt injection detected."
        reason += f" Risk Analysis: {node_input.risk_analysis}"

        yield Event(
            output={
                "status": "rejected",
                "reason": reason,
                "amount": ctx.state.get("amount"),
            },
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=f"Expense of ${ctx.state.get('amount')} was rejected by human reviewer. {reason}"
                    )
                ],
            ),
        )


# ------------------------------------------------------------------------------
# 3. Workflow Graph Construction
# ------------------------------------------------------------------------------

root_agent = Workflow(
    name="expense_triage_workflow",
    input_schema=EventInput,
    edges=[
        (START, parse_expense_event),
        (parse_expense_event, route_expense),
        # Route from route_expense using RoutingMap dict
        (route_expense, {"low_value": auto_approve, "high_value": security_screen}),
        # Route from security_screen using RoutingMap dict
        (
            security_screen,
            {"security_passed": review_agent, "security_flagged": human_review_pause},
        ),
        # Unconditional connection from review_agent to human_review_pause
        (review_agent, human_review_pause),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
