# IBM Cloud & watsonx.ai Setup Guide

Step-by-step guide to enable real Granite AI reasoning in MissionMind.

## Prerequisites

- An email address for IBM Cloud account creation
- A modern web browser

## Step 1: Create IBM Cloud Account

1. Go to [cloud.ibm.com/registration](https://cloud.ibm.com/registration)
2. Click **Create a free account** (Lite plan — no credit card required)
3. Fill in your details and verify your email
4. Log in to the IBM Cloud dashboard

## Step 2: Enable watsonx.ai

1. From the IBM Cloud dashboard, search for **watsonx.ai** in the catalog
2. Click **Create** to provision a watsonx.ai instance (Lite plan is free)
3. Wait for the instance to become **Active** (usually 1-2 minutes)

## Step 3: Create a Project

1. Open the watsonx.ai console: [watsonx.ai](https://watsonx.ai)
2. Click **Projects** → **New project**
3. Name it (e.g., `MissionMind`)
4. Associate it with your watsonx.ai service instance
5. Click **Create**

## Step 4: Generate an API Key

1. Go to [IBM Cloud API Keys](https://cloud.ibm.com/iam/apikeys)
2. Click **Create**
3. Name it (e.g., `missionmind-watsonx`)
4. Copy the key immediately — it won't be shown again

## Step 5: Get Your Project ID

1. In the watsonx.ai console, open your project
2. The project ID is in the URL: `watsonx.ai/my-project/<PROJECT_ID>/assets`
3. Or click **Manage** → **General** → **Project ID**

## Step 6: Configure MissionMind

Create a `.env` file in the project root:

```bash
WATSONX_APIKEY=your-api-key-here
WATSONX_PROJECT_ID=your-project-id-here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

## Step 7: Verify

```bash
# This makes ONE real call to IBM Granite
python -m missionmind.ai.granite_client --check
```

Expected output:
```
CHECK PASS — Granite answered from ibm/granite-4-h-small
```

If it says `CHECK FAIL`, verify your API key and project ID.

## Step 8: Run the Dashboard with Real Granite

```bash
# Restart the API server with credentials loaded
python -m uvicorn missionmind.viz.api_server:app --port 8100
```

The dashboard sidebar will show **API Key: configured (real Granite)** instead of `mock fallback`.

## Troubleshooting

| Issue | Fix |
|---|---|
| `WATSONX_APIKEY not set` | Check `.env` file exists in project root with correct key |
| `CHECK FAIL` | API key may be expired — regenerate at cloud.ibm.com/iam/apikeys |
| `Model not found` | Ensure watsonx.ai instance is **Active** (not provisioning) |
| `Project not found` | Verify project ID matches the watsonx.ai project URL |
| Rate limit errors | Lite plan has rate limits — wait 30s and retry |

## What Happens Without Credentials

MissionMind works fully without IBM credentials. The Granite client falls back to a deterministic mock that returns the same JSON schema with RAG citations. The dashboard clearly shows which mode is active:

- **No credentials**: `API Key: missing (mock fallback)` — deterministic reasoning
- **With credentials**: `API Key: configured (real Granite)` — AI-generated reasoning

Both modes are honest — the UI never disguises the mock as real.

## Cost

The watsonx.ai Lite plan includes:
- Free tier with generous monthly limits
- `ibm/granite-4-h-small` is included in the free tier
- No credit card required for Lite plan
