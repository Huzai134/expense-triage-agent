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

import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes
from app.app_utils.telemetry import setup_telemetry
from app.app_utils.typing import Feedback

# Setup standard logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
setup_telemetry()

allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.agent import app as adk_app
    from app.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=False,
    lifespan=lifespan,
)
app.title = "expense-triage-agent"
app.description = "API for interacting with the Agent expense-triage-agent"


class PubSubMessage(BaseModel):
    data: Any
    message_id: str = Field(alias="messageId")
    publish_time: str = Field(alias="publishTime", default="")

    model_config = ConfigDict(populate_by_name=True)


class PubSubPushRequest(BaseModel):
    message: PubSubMessage
    subscription: str


@app.post("/pubsub")
async def handle_pubsub(request: PubSubPushRequest):
    logger.info(
        f"Received Pub/Sub message ID {request.message.message_id} on subscription {request.subscription}"
    )

    # Normalize subscription path to keep session records readable
    subscription_path = request.subscription
    normalized_sub = subscription_path.split("/")[-1]

    runner: Runner = app.state.runner
    session_id = f"{normalized_sub}-{request.message.message_id}"
    user_id = normalized_sub

    # Get or create session
    session = await runner.session_service.get_session(
        app_name=app.state.agent_app_name, user_id=user_id, session_id=session_id
    )
    if not session:
        session = await runner.session_service.create_session(
            app_name=app.state.agent_app_name, user_id=user_id, session_id=session_id
        )

    # Prepare input for workflow
    input_payload = {"data": request.message.data}
    new_message = types.Content(
        role="user", parts=[types.Part.from_text(text=json.dumps(input_payload))]
    )

    # Execute workflow
    events = []
    async for event in runner.run_async(
        new_message=new_message, user_id=user_id, session_id=session.id
    ):
        events.append(event)

    logger.info(
        f"Workflow completed for session {session.id}. Processed {len(events)} events."
    )

    return {
        "status": "success",
        "subscription": normalized_sub,
        "message_id": request.message.message_id,
        "session_id": session.id,
        "events_count": len(events),
    }


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.info(f"Feedback: {feedback.model_dump()}")
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
