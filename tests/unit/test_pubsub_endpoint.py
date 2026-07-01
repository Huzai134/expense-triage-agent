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

from fastapi.testclient import TestClient

from app.fast_api_app import app

client = TestClient(app)


def test_pubsub_endpoint_base64() -> None:
    """Test the /pubsub webhook endpoint with base64-encoded payload."""
    raw_data = {
        "amount": 45.0,
        "submitter": "Alice",
        "category": "Meals",
        "description": "Lunch with client",
        "date": "2026-07-01",
    }
    encoded_data = base64.b64encode(json.dumps(raw_data).encode("utf-8")).decode(
        "utf-8"
    )

    payload = {
        "message": {
            "data": encoded_data,
            "messageId": "msg-12345",
        },
        "subscription": "projects/my-project/subscriptions/test-sub-base64",
    }

    with TestClient(app) as client:
        response = client.post("/pubsub", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["subscription"] == "test-sub-base64"
        assert data["message_id"] == "msg-12345"
        assert "session_id" in data
        assert data["events_count"] > 0


def test_pubsub_endpoint_plain() -> None:
    """Test the /pubsub webhook endpoint with plain JSON payload."""
    raw_data = {
        "amount": 150.0,
        "submitter": "Bob",
        "category": "Equipment",
        "description": "New developer monitor",
        "date": "2026-07-01",
    }

    payload = {
        "message": {
            "data": raw_data,
            "messageId": "msg-67890",
        },
        "subscription": "projects/my-project/subscriptions/test-sub-plain",
    }

    with TestClient(app) as client:
        response = client.post("/pubsub", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["subscription"] == "test-sub-plain"
        assert data["message_id"] == "msg-67890"
        assert "session_id" in data
        assert data["events_count"] > 0
