# Deploy MissionMind to Vercel

Step-by-step guide to deploy the React web console + FastAPI backend to Vercel.

## Prerequisites

- A GitHub account with the repo pushed
- A Vercel account (free tier works)

## Step 1: Import the Repo

1. Go to [vercel.com/new](https://vercel.com/new)
2. Click **Import Git Repository**
3. Select `ojasvigoel598/IBM-spacecraft`
4. Click **Import**

## Step 2: Configure the Project

Vercel should auto-detect the configuration from `vercel.json`:

| Setting | Value |
|---|---|
| Framework Preset | **Other** |
| Build Command | `npm --prefix web run build` |
| Install Command | `npm --prefix web install` |
| Output Directory | `web/dist` |
| Root Directory | (leave blank — project root) |

If Vercel doesn't detect these automatically, set them manually in the
**Build & Development Settings** section.

## Step 3: Add Environment Variables

In the **Environment Variables** section, add:

| Variable | Value | Notes |
|---|---|---|
| `MISSIONMIND_ENV` | `production` | Tells the app it's in prod mode |
| `MISSIONMIND_DB_PATH` | `/tmp/missionmind.db` | SQLite on Vercel's ephemeral filesystem |
| `MISSIONMIND_SECRET_KEY` | (generate a random string) | Session encryption key |
| `MISSIONMIND_ALLOWED_ORIGINS` | `https://your-project.vercel.app` | CORS origins (set after deploy) |

Optionally, if you have IBM watsonx.ai credentials:

| Variable | Value |
|---|---|
| `WATSONX_APIKEY` | Your IBM Cloud API key |
| `WATSONX_PROJECT_ID` | Your watsonx.ai project ID |
| `WATSONX_URL` | `https://us-south.ml.cloud.ibm.com` |

**Important:** Vercel's serverless functions use an ephemeral filesystem.
SQLite data resets on each cold start. For persistent data, you'd need
an external database (Supabase, Turso, etc.). The current implementation
is fine for demo/hackathon purposes — the app regenerates data on startup.

## Step 4: Deploy

1. Click **Deploy**
2. Wait for the build to complete (~1-2 minutes)
3. Vercel will give you a URL like `https://ibm-spacecraft-xxxx.vercel.app`

## Step 5: Update CORS

After the first deploy, update the `MISSIONMIND_ALLOWED_ORIGINS` environment
variable to include your actual Vercel URL:

```
https://ibm-spacecraft-xxxx.vercel.app
```

Then redeploy (Settings → Deployments → Redeploy).

## Step 6: Verify

1. Open your Vercel URL
2. You should see the MissionMind web console (React + Vite)
3. Register a test account or use the default dev credentials
4. The sidebar should show connection status to the API

## Architecture on Vercel

```text
Browser → Vercel Edge Network
    ├── /           → React SPA (static, from web/dist)
    ├── /api/*      → Python serverless function (FastAPI)
    └── /assets/*   → Static assets (JS, CSS, images)
```

The FastAPI backend runs as a Vercel Python serverless function.
The React frontend is a static SPA served from the CDN.

**Note:** The Streamlit dashboard (port 8501) does NOT run on Vercel.
It's a separate Python server app for local development and the
3D spacecraft visualization.

## Troubleshooting

| Issue | Fix |
|---|---|
| Build fails with `npm not found` | Ensure `vercel.json` has `installCommand: "npm --prefix web install"` |
| API returns 500 | Check function logs in Vercel dashboard → Functions tab |
| CORS errors | Set `MISSIONMIND_ALLOWED_ORIGINS` to your Vercel URL |
| Cold start slow | First request after idle takes ~5s (normal for serverless Python) |
| Data disappears | SQLite is ephemeral on Vercel — expected behavior |

## Updating the Deploy

After pushing to `main`, Vercel auto-deploys. To deploy manually:

```bash
npx vercel --prod
```

Or go to the Vercel dashboard → Deployments → Redeploy.
