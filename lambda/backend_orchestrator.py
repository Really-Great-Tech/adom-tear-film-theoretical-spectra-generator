"""
Lambda: PyElli backend EC2 orchestrator.

Two modes (via event["action"]):
  - "start"   : Start the EC2 instance if stopped. Called by Streamlit (or API Gateway)
                when user triggers grid search / full test and backend is cold.
  - "check_idle": Call backend GET /api/last-activity; if idle longer than IDLE_TIMEOUT_SECONDS,
                  stop the EC2 instance. Invoked by EventBridge on a schedule (e.g. every 5 min).

Required environment variables (set in Lambda configuration):
  - INSTANCE_ID       : EC2 instance ID (e.g. i-0abc123...)
  - BACKEND_BASE_URL  : Base URL of the backend when EC2 is running (e.g. http://10.0.1.50:8000
                        or https://backend.example.com). Used for /health and /api/last-activity.

Optional:
  - IDLE_TIMEOUT_SECONDS : Seconds of inactivity before stopping EC2 (default 900 = 15 min).
  - AWS_REGION         : Region for EC2 (default: same as Lambda).

IAM: Lambda needs ec2:DescribeInstances, ec2:DescribeInstanceStatus, ec2:StartInstances, ec2:StopInstances.
"""

import json
import os
import time
from typing import Any, Dict

import boto3
import requests

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

INSTANCE_ID = os.environ.get("INSTANCE_ID", "").strip()
BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "").strip().rstrip("/")
IDLE_TIMEOUT_SECONDS = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "900"))
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", ""))

# ---------------------------------------------------------------------------
# EC2 helpers
# ---------------------------------------------------------------------------


def get_ec2():
    return boto3.client("ec2", region_name=AWS_REGION or None)


def get_instance_state(instance_id: str) -> str:
    """Return instance state: pending, running, stopping, stopped, etc."""
    ec2 = get_ec2()
    r = ec2.describe_instances(InstanceIds=[instance_id])
    for res in r.get("Reservations", []):
        for inst in res.get("Instances", []):
            return inst["State"]["Name"]
    return "unknown"


def start_instance(instance_id: str) -> Dict[str, Any]:
    ec2 = get_ec2()
    ec2.start_instances(InstanceIds=[instance_id])
    return {"instance_id": instance_id, "action": "start_instances"}


def stop_instance(instance_id: str) -> Dict[str, Any]:
    ec2 = get_ec2()
    ec2.stop_instances(InstanceIds=[instance_id])
    return {"instance_id": instance_id, "action": "stop_instances"}


# ---------------------------------------------------------------------------
# Backend HTTP helpers
# ---------------------------------------------------------------------------


def backend_health_ok() -> bool:
    if not BACKEND_BASE_URL:
        return False
    try:
        r = requests.get(f"{BACKEND_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def get_last_activity() -> float | None:
    """Return backend last_activity epoch float, or None if unreachable."""
    if not BACKEND_BASE_URL:
        return None
    try:
        r = requests.get(f"{BACKEND_BASE_URL}/api/last-activity", timeout=5)
        if r.status_code != 200:
            return None
        return float(r.json().get("last_activity", 0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    event["action"] (or event["body"] parsed from API Gateway/Function URL):
      - "start"      : Start EC2 if stopped; return status.
      - "check_idle" : If backend idle longer than IDLE_TIMEOUT_SECONDS, stop EC2.
    """
    # Support both direct invoke (event = {"action": "start"}) and API Gateway/Function URL (event["body"] = '{"action":"start"}')
    body = event.get("body")
    if isinstance(body, str):
        try:
            event = {**event, **json.loads(body)}
        except json.JSONDecodeError:
            pass

    def resp(code: int, body: Dict[str, Any]) -> Dict[str, Any]:
        return {"statusCode": code, "body": json.dumps(body)}

    if not INSTANCE_ID:
        return resp(500, {"error": "INSTANCE_ID not configured", "action": event.get("action")})

    action = (event.get("action") or event.get("Action") or "start").strip().lower()

    if action == "start":
        return handle_start(event, resp)
    if action == "check_idle":
        return handle_check_idle(event, resp)
    return resp(400, {"error": f"Unknown action: {action}", "allowed": ["start", "check_idle"]})


def handle_start(event: Dict[str, Any], resp: Any) -> Dict[str, Any]:
    """Start EC2 if stopped. Streamlit will poll /health until backend is up."""
    state = get_instance_state(INSTANCE_ID)
    if state == "running":
        return resp(200, {"status": "running", "message": "Instance already running", "instance_id": INSTANCE_ID})
    if state == "stopped":
        start_instance(INSTANCE_ID)
        return resp(200, {
            "status": "starting",
            "message": "EC2 start requested; backend may take 1–2 min to become healthy",
            "instance_id": INSTANCE_ID,
        })
    if state in ("pending", "stopping"):
        return resp(200, {"status": state, "message": f"Instance is {state}; wait and retry /health", "instance_id": INSTANCE_ID})
    return resp(500, {"error": f"Unexpected instance state: {state}", "instance_id": INSTANCE_ID})


def handle_check_idle(event: Dict[str, Any], resp: Any) -> Dict[str, Any]:
    """If backend last_activity is older than IDLE_TIMEOUT_SECONDS, stop EC2."""
    last = get_last_activity()
    now = time.time()
    if last is None:
        return resp(200, {"action": "check_idle", "stopped": False, "message": "Backend unreachable; no action taken", "instance_id": INSTANCE_ID})
    idle_seconds = now - last
    if idle_seconds < IDLE_TIMEOUT_SECONDS:
        return resp(200, {
            "action": "check_idle",
            "stopped": False,
            "idle_seconds": round(idle_seconds, 1),
            "threshold_seconds": IDLE_TIMEOUT_SECONDS,
            "message": f"Backend still active (idle {idle_seconds:.0f}s < {IDLE_TIMEOUT_SECONDS}s)",
            "instance_id": INSTANCE_ID,
        })
    state = get_instance_state(INSTANCE_ID)
    if state != "running":
        return resp(200, {"action": "check_idle", "stopped": False, "message": f"Instance not running (state={state}); no stop needed", "instance_id": INSTANCE_ID})
    stop_instance(INSTANCE_ID)
    return resp(200, {
        "action": "check_idle",
        "stopped": True,
        "idle_seconds": round(idle_seconds, 1),
        "threshold_seconds": IDLE_TIMEOUT_SECONDS,
        "message": f"Stopped instance after {idle_seconds:.0f}s idle",
        "instance_id": INSTANCE_ID,
    })
