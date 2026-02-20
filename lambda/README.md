# PyElli Backend Orchestrator (Lambda)

This Lambda starts the PyElli backend EC2 instance on demand (cold start) and stops it after a period of inactivity. The Streamlit app (ECS) calls the Lambda when the user runs grid search or full test and the backend is not reachable.

## What the Lambda does

| Action | Who calls it | Behaviour |
|--------|----------------|-----------|
| **start** | Streamlit app (or API Gateway) when user triggers heavy work | If EC2 is stopped → start it. Returns immediately; the app then polls backend `/health` until ready. |
| **check_idle** | EventBridge schedule (e.g. every 5 min) | Calls backend `GET /api/last-activity`. If last activity &gt; 15 min ago → stop EC2. |

## Deployment (DevOps)

### 1. Create the Lambda function

- **Runtime:** Python 3.11 (or 3.12)
- **Handler:** `backend_orchestrator.lambda_handler`
- **Code:** Zip and upload the contents of this `lambda/` folder (at least `backend_orchestrator.py`). If you use a layer or inline dependencies, include `requests` (boto3 is already in the runtime).

To build a deployment package:

```bash
cd lambda
pip install -r requirements.txt -t package/
cp backend_orchestrator.py package/
cd package && zip -r ../lambda_backend_orchestrator.zip . && cd ..
# Upload lambda_backend_orchestrator.zip as the Lambda code.
```

### 2. Environment variables (required)

| Variable | Description | Example |
|----------|-------------|---------|
| **INSTANCE_ID** | EC2 instance ID of the PyElli backend | `i-0abc123def456` |
| **BACKEND_BASE_URL** | Base URL of the backend when EC2 is running. Used for `/health` and `/api/last-activity`. Must be reachable from the Lambda (same VPC or public). | `http://10.0.1.50:8000` or `https://backend.example.com` |

### 3. Environment variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| **IDLE_TIMEOUT_SECONDS** | `900` (15 min) | Idle time after which EC2 is stopped by `check_idle`. |
| **AWS_REGION** | (Lambda default) | Region for EC2 API calls. |

### 4. IAM permissions

Attach a policy that allows:

- `ec2:DescribeInstances`
- `ec2:DescribeInstanceStatus`
- `ec2:StartInstances`
- `ec2:StopInstances`

Resource: the specific instance (or `"Resource": "arn:aws:ec2:<region>:<account>:instance/<INSTANCE_ID>"`).

A minimal policy is in **`iam-policy-example.json`** (allows describe/start/stop for EC2). You can restrict `Resource` to the specific instance ARN for tighter security.

### 5. Expose “start” to the Streamlit app (Lambda Function URL)

- In the Lambda console, add a **Function URL** (e.g. auth type NONE for a simple POST from ECS).
- Copy the URL and set it in the **ECS task definition** (Streamlit app) as:

  `BACKEND_START_LAMBDA_URL=<that URL>`

The app sends:

```http
POST <BACKEND_START_LAMBDA_URL>
Content-Type: application/json

{"action": "start"}
```

So the Function URL must accept POST with this body.

### 6. Schedule “check_idle” (EventBridge)

- Create an EventBridge rule: schedule expression `rate(5 minutes)` (or `cron(0/5 * * * ? *)`).
- Target: this Lambda.
- **Important:** Set the target’s **input** to a constant JSON so the Lambda runs the idle check:  
  `{"action": "check_idle"}`  
  (If you don’t pass this, the Lambda defaults to `"start"`, which only starts the instance and is not what you want on a schedule.)

### 7. Network (BACKEND_BASE_URL)

- If the backend is in a **private VPC**, put the Lambda in the **same VPC** (and subnets that can reach the EC2) and set **BACKEND_BASE_URL** to the EC2 private IP or an internal ALB/DNS (e.g. `http://10.0.1.50:8000`).
- If the backend has a **public IP or ALB**, you can use that URL and run the Lambda without VPC (or in VPC with NAT) so it can reach the internet.

## Summary checklist for DevOps

1. Create Lambda from `lambda/backend_orchestrator.py` (and optional `requirements.txt`).
2. Set env: **INSTANCE_ID**, **BACKEND_BASE_URL**; optionally **IDLE_TIMEOUT_SECONDS**, **AWS_REGION**.
3. Attach IAM policy for EC2 describe/start/stop.
4. Add **Lambda Function URL**; set **BACKEND_START_LAMBDA_URL** in ECS (Streamlit) to that URL.
5. Add **EventBridge** rule every 5 min targeting this Lambda with payload `{"action": "check_idle"}`.

After that, the Streamlit app can wake the backend on demand, and the backend will shut down after 15 minutes (or your configured idle time) of no requests.
