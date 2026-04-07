---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - email-triage
  - nlp
  - classification
  - rl-environment
license: mit
short_description: Real-world OpenEnv environment for AI email triage agent training
---

# 📧 Email Triage OpenEnv v2

A production-ready **OpenEnv** environment where AI agents learn to classify, prioritize,
tag, and route real business emails — exactly as a human operations team does.

## ⚡ Quickstart

```python
import httpx

BASE = "https://your-space.hf.space"

# 1. Start a session and get a session_id
r = httpx.post(f"{BASE}/reset", json={"task_id": "easy_triage"})
session_id = r.json()["session_id"]   # ← required for all further calls
obs = r.json()["observation"]

# 2. Step through the episode until done
while not obs["done"]:
    action = {
        "email_id": obs["email_id"],
        "category": "support",
        "priority": "high",
        "route_to": "tech_team",
        "tags": ["login", "account_access"],
        "notes": "Login issue reported"
    }
    r = httpx.post(f"{BASE}/step?session_id={session_id}", json={"action": action})
    data = r.json()
    obs  = data["observation"]
    print(f"Step score: {data['reward']['step_score']:.4f}")

# 3. Final grader score
print(f"Episode done. Cumulative: {data['reward']['cumulative_score']:.4f}")
```

## 🎮 Tasks

| Task ID | Emails | Difficulty | Pass Threshold |
|---------|--------|------------|----------------|
| `easy_triage` | 3 | 🟢 Easy | 0.80 |
| `medium_triage` | 5 | 🟡 Medium | 0.75 |
| `hard_triage` | 10 | 🔴 Hard | 0.65 |

## 🔌 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start episode → returns `session_id` |
| `POST` | `/step?session_id=...` | Submit one action → observation + reward |
| `GET` | `/state?session_id=...` | Inspect current episode state |
| `GET` | `/tasks` | List tasks + full action schema |
| `POST` | `/grader` | Offline deterministic grading |
| `POST` | `/baseline` | Run GPT-4o-mini baseline (needs API key) |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive Swagger UI |

## 🔑 Setting OPENAI_API_KEY

The `/baseline` endpoint requires an OpenAI key. Add it as a **Space Secret** in your
HF Space Settings → Variables and secrets → New secret named `OPENAI_API_KEY`.

## 📊 Baseline Results (GPT-4o-mini, temperature=0)

| Task | Score | Threshold | Status |
|------|-------|-----------|--------|
| easy_triage | 0.9167 | 0.80 | ✅ PASSED |
| medium_triage | 0.7886 | 0.75 | ✅ PASSED |
| hard_triage | TBD (v3 upgrade) | 0.65 | Re-run baseline |

Average: **0.7887** (easy + medium) — hard task re-run needed after v3 upgrade

## 🏗️ Reward Design

- **Partial credit**: priority within 1 level → 0.5, exact → 1.0
- **Tag F1 scoring**: precision-recall F1 on tag sets
- **6 penalty types**: invalid ID, duplicate, over-steps, empty tags, contradictory route, SLA breach
- **5 bonus types**: speed, extra-tag, sentiment-aware, SLA-aware, escalation-aware

## 📖 Full Docs

See the [README.md](README.md) for complete documentation including full task descriptions,
grader weights, and environment design rationale.
