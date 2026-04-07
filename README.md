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

# 📧 Email Triage OpenEnv  

> **A production-ready OpenEnv environment** where AI agents learn to read, classify, prioritize, and route real-world business emails — just like a human operations team does every day.

---

## 🌍 Why Email Triage Matters in the Real World

Every company — from a 5-person startup to a Fortune 500 enterprise — receives hundreds of emails daily. These emails arrive in a single shared inbox and must be:

- **Classified** (Is this a billing issue? A legal threat? A sales opportunity?)
- **Prioritized** (Does this need a response in 5 minutes or 5 days?)
- **Tagged** (What topics does it cover, for future search and analytics?)
- **Routed** (Which team is responsible — tech support, billing, legal, HR, sales?)

Today, **human agents spend 30–40% of their working day** on this manual triage work. It is slow, inconsistent, and expensive. Companies like Zendesk, Freshdesk, Intercom, and Salesforce are actively investing in AI-based inbox management.

This environment simulates exactly that real-world workflow. By training and evaluating AI agents here, we move toward:

- **Faster response times** for customers
- **Consistent classification** across all email types
- **Reduced cost** of operations teams
- **Better agent specialization** — tech agents get only tech emails, legal gets only legal, etc.

---

## 🤖 What Does the AI Agent Learn?

An agent that masters this environment develops the following capabilities:

| Capability | What it means |
|---|---|
| **Intent recognition** | Understand why someone is writing (complaint? inquiry? emergency?) |
| **Urgency detection** | Distinguish truly urgent situations from low-priority requests |
| **Context understanding** | Read between the lines — a polite email can still be high-stakes |
| **Entity classification** | Identify legal, HR, sales, tech content correctly |
| **Risk awareness** | Recognize data breaches, harassment, outages as critical |
| **Multi-label reasoning** | Apply multiple correct tags simultaneously |

This environment is deliberately designed to **require reasoning**, not just keyword matching. For example:

- A data breach email might sound like a "support" issue — but it is **legal + urgent**
- A "partnership opportunity" email might look like spam — but it is **sales + high priority**
- A friendly "can you do a report for me?" email is **support + low priority**, not sales

---

## 👁️ Observation Space

At each step, the agent receives one `Observation` representing the current email:

| Field | Type | Description |
|---|---|---|
| `email_id` | str | Unique email identifier |
| `subject` | str | Email subject line |
| `body` | str | Full email body (may include thread context) |
| `sender` | str | Sender's email address |
| `timestamp` | str | ISO 8601 timestamp |
| `step` | int | Current step (0-indexed) |
| `total_steps` | int | Total emails in this task |
| `task_id` | str | Active task ID |
| `history` | list | All previous actions taken this episode |
| `done` | bool | True if episode is complete |
| `message` | str | Contextual message from environment |
| `sla_hours_remaining` | float \| null | Estimated hours until deadline (e.g. `3.0` for "presentation in 3h"). `null` = no deadline detected. Treat `< 2` → URGENT, `< 24` → HIGH |

> **Thread Context**: Some emails include a `📎 THREAD CONTEXT` prefix in the body — this simulates a reply chain and gives the agent extra information to reason with. For example, a follow-up refund email includes context that the first request was sent 7 days ago with no response.

---

## 🎯 Action Space

For each email, the agent submits an `Action`:

| Field | Type | Allowed Values |
|---|---|---|
| `email_id` | str | Must match the current email's ID |
| `category` | enum | `spam` `support` `billing` `sales` `hr` `legal` `general` |
| `priority` | enum | `low` `medium` `high` `urgent` |
| `route_to` | enum | `sales_team` `tech_team` `billing_team` `hr_team` `legal_team` `trash` `inbox` |
| `tags` | list[str] | 2–4 keyword tags, e.g. `["refund", "angry_customer"]` |
| `notes` | str | Optional agent reasoning text |

---

## 🔄 Example: Full Agent Interaction

A complete turn-by-turn example using the `easy_triage` task.

### Step 1 — Reset (`POST /reset`)

**Request:**
```json
{ "task_id": "easy_triage" }
```

**Response:**
```json
{
  "session_id": "a3f8c1d2-...",
  "task_info": { "task_id": "easy_triage", "email_count": 3, "pass_threshold": 0.8 },
  "observation": {
    "email_id": "easy_001",
    "subject": "YOU WON $1,000,000 CLICK NOW!!!",
    "sender": "prize_winner@totally-legit-lottery.ru",
    "body": "Congratulations! You have been selected as our lucky winner...",
    "step": 0,
    "total_steps": 3,
    "sla_hours_remaining": null,
    "done": false
  }
}
```

### Step 2 — Submit Action (`POST /step?session_id=a3f8c1d2-...`)

**Request:**
```json
{
  "action": {
    "email_id": "easy_001",
    "category": "spam",
    "priority": "low",
    "route_to": "trash",
    "tags": ["spam", "phishing"],
    "notes": "Lottery scam from .ru domain"
  }
}
```

**Response:**
```json
{
  "observation": {
    "email_id": "easy_002",
    "subject": "My account login is not working",
    "sla_hours_remaining": null,
    "step": 1,
    "total_steps": 3,
    "done": false
  },
  "reward": {
    "step_score": 1.0,
    "cumulative_score": 1.0,
    "penalty": 0.0,
    "breakdown": { "category": 1.0, "priority": 1.0, "route": 1.0, "tags": 1.0 },
    "feedback": "✅ Category 'spam' correct. ✅ Priority 'low' correct. ✅ Route 'trash' correct."
  },
  "done": false,
  "info": { "step": 1, "emails_remaining": ["easy_002", "easy_003"], "terminal": false }
}
```

### SLA Example — Hard task email with deadline

When the agent receives `hard_003` ("subscription renewal failed + client presentation in 3h"), the observation includes:

```json
{
  "email_id": "hard_003",
  "subject": "Your subscription renewal failed",
  "sla_hours_remaining": 3.0,
  "body": "...I need access restored urgently because I have a client presentation in 3 hours..."
}
```

If the agent assigns `priority: "low"` despite `sla_hours_remaining: 3.0`, it receives a **−0.10 SLA breach penalty** in addition to the normal priority scoring penalty.

---

## 🎮 Tasks — 3 Difficulty Levels

### 🟢 Easy — `easy_triage`

**3 emails.** Clear language. No ambiguity. Tests basic classification.

| Email | Content | Expected | Difficulty Reason |
|---|---|---|---|
| easy_001 | "YOU WON $1,000,000!!!" lottery scam | spam → trash | Obviously spam |
| easy_002 | "My login doesn't work since yesterday" | support → tech_team → high | Clear support issue |
| easy_003 | Invoice #INV-4821 payment due | billing → billing_team → medium | Straightforward billing |

- **Grading weights**: category 50% · priority 25% · route 25%
- **Pass threshold**: 0.80

---

### 🟡 Medium — `medium_triage`

**5 emails.** Mix of urgency levels. Some ambiguity. Tests prioritization.

| Email | Content | Key Challenge |
|---|---|---|
| med_001 | Production server down, $5000/min loss | Must detect URGENT (not just high) |
| med_002 | Casual pricing question, "no rush" | Must detect LOW urgency despite sales context |
| med_003 | Follow-up refund request (7 days ignored) | Thread context: escalating → HIGH |
| med_004 | Newsletter unsubscribe request | Must recognize as GENERAL → inbox → low |
| med_005 | B2B partnership inquiry, 50K users | Must recognize business value → HIGH |

- **Grading weights**: category 35% · priority 35% · route 20% · tags 10%
- **Pass threshold**: 0.75

---

### 🔴 Hard — `hard_triage`

**10 emails.** Complex, adversarial, and deliberately deceptive. Requires full contextual, sentiment, and domain reasoning. Designed to challenge frontier models.

| Email | Content | Why It's Hard |
|---|---|---|
| hard_001 | Data breach — 2000 user records exposed | Sounds like IT/support but it's **LEGAL + URGENT** (GDPR 72h rule) |
| hard_002 | Employee harassment complaint | Looks like support but it's **HR + HIGH** + confidential |
| hard_003 | Subscription failed + client presentation in 3h | **BILLING + URGENT** — time pressure explicit |
| hard_004 | Contract liability dispute — Section 4.2 | **LEGAL + HIGH** — not sales. Contract expiry June 10 |
| hard_005 | Friendly custom report request | Polite → **SUPPORT + LOW**. Not sales |
| hard_006 | Job application — Senior Data Analyst | **HR + LOW** — not support |
| hard_007 | 500-license enterprise deal, VP decision maker | **SALES + HIGH** — procurement deadline June 20 |
| hard_008 🆕 | "Small display bug?" — actually a double charge | **BILLING + HIGH** — polite tone hides a real billing error |
| hard_009 🆕 | Enterprise prospect needs DPA signed first | **LEGAL + HIGH** — not sales. GDPR Article 28 blocks commercial deal |
| hard_010 🆕 | Fake internal IT security notice | **SPAM + TRASH + LOW** — convincing phishing from lookalike domain |

**Adversarial design rationale:**
- `hard_008`: Customer uses "display bug?" framing to stay polite — agent must see through the friendly tone to the real billing dispute
- `hard_009`: Looks like a sales lead — but a DPA is a legal document and must go to legal *first*. Sales cannot proceed without it
- `hard_010`: Spoofed domain (`ourcompany-systems.xyz` vs real `ourcompany.com`), credential-harvesting link, urgent-sounding IT language — classic spear-phishing

- **Grading weights**: category 25% · priority 25% · route 25% · tags 25%
- **Pass threshold**: 0.65

---

## 🏆 Reward Function (v2)

### Base Score Formula

```
step_score = (category_score  × 0.50)
           + (priority_score  × 0.20)
           + (route_score     × 0.20)
           + (tag_score       × 0.10)
```

### Priority Scoring (Partial Credit)

| Gap between predicted and correct | Score |
|---|---|
| Exact match | 1.00 |
| 1 level off (e.g., HIGH instead of URGENT) | 0.50 |
| 2+ levels off | 0.00 |

### Tag Scoring (F1)

Tags are scored using precision-recall F1 between predicted tags and expected tags. Partial matches receive partial credit.

### Bonuses

| Bonus | Condition | Amount |
|---|---|---|
| Speed bonus | Processed email correctly before halfway point | +0.05 |
| Extra tag bonus | Found additional correct tags beyond minimum | +0.05 |
| **Sentiment bonus** 🆕 | Email contains negative sentiment AND agent assigns high/urgent (correctly) | +0.03 |
| **SLA bonus** 🆕 | Email contains time deadline AND agent assigns high/urgent (correctly) | +0.03 |
| **Escalation bonus** 🆕 | Email contains escalation threat AND agent assigns high/urgent (correctly) | +0.03 |

> The three new bonuses in v3 reward agents that reason *why* an email is urgent, not just *that* it is urgent. An agent that detects `"I'm getting frustrated"` or `"I may escalate to my bank"` and responds with high priority earns extra reward beyond the base classification score.

### Penalties

| Penalty | Condition | Amount |
|---|---|---|
| Invalid email ID | Email ID does not exist in the task | −0.30 |
| Duplicate action | Same email processed twice | −0.20 |
| Over step limit | Exceeded max_steps for task | −0.10 |
| Empty tags | No tags provided on hard_triage | −0.05 |
| Contradictory routing | e.g., spam → billing_team | −0.05 |
| **SLA breach** 🆕 | `sla_hours_remaining < 2` but agent assigned LOW or MEDIUM priority | −0.10 |
| **Loop / over-steps** | Agent exceeds `max_steps` for the task (episode force-terminated) | −0.10 |

> All final scores are clamped to [0.0, 1.0]

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Start new episode — returns `session_id` |
| POST | `/step?session_id=...` | Submit action, get reward |
| GET | `/state?session_id=...` | Inspect current episode state |
| GET | `/tasks` | List all 3 tasks |
| POST | `/grader` | Offline deterministic grading |
| POST | `/baseline` | Run GPT-4o-mini agent |
| GET | `/sessions` | List active sessions |
| DELETE | `/sessions/{id}` | Delete a session |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

### Multi-Session Design

Each call to `/reset` returns a unique `session_id`. Pass this in subsequent `/step` and `/state` calls. Multiple agents or users can run simultaneously without interfering with each other.

---

## 🚀 Setup & Run

### Local Python

```bash
# 1. Navigate to project
cd email_triage_env

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# 5. Open docs
# → http://localhost:7860/docs
```

### Docker

```bash
# Build
docker build -t email-triage-env .

# Run
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... email-triage-env

# → http://localhost:7860/docs
```

### Hugging Face Spaces

1. Create new Space → select **Docker** SDK
2. Upload all project files
3. Add `OPENAI_API_KEY` as a **Secret** in Space Settings
4. App auto-deploys on port 7860

---

## 🧪 Run Tests

```bash
pytest tests/test_env.py -v
# → 98 passed ✅
```

---

## 🤖 Run Baseline Agent

```bash
export OPENAI_API_KEY="sk-..."
python scripts/baseline.py          # all 3 tasks
python scripts/baseline.py --task easy_triage
python scripts/baseline.py --quiet  # summary only
```

### Expected Baseline Scores (GPT-4o-mini, temperature=0)

| Task | Expected Score | Threshold | Expected Status |
|---|---|---|---|
| easy_triage | 0.92 – 0.98 | 0.80 | ✅ PASS |
| medium_triage | 0.80 – 0.90 | 0.75 | ✅ PASS |
| hard_triage | 0.55 – 0.68 | 0.65 | ⚠️ Borderline — adversarial emails challenge GPT-4o-mini |

> Hard task was upgraded to 10 emails (3 adversarial additions). The lower pass threshold (0.65) reflects the increased difficulty while keeping the task meaningful for RL training.

> Scores are fully reproducible (temperature=0). Results saved to `baseline_results.json`.

---

## 🧠 Architecture

```
Agent / User
    │
    ├── POST /reset ─────► get_or_create_session()
    │                           └── EmailTriageEnv.reset(task_id)
    │                                   └── loads emails + ground truth
    │                                   └── returns Observation (first email)
    │
    ├── POST /step ──────► EmailTriageEnv.step(action)
    │                           ├── compute_reward()
    │                           │       ├── grade_single_action() → weighted score
    │                           │       ├── apply penalties (5 types)
    │                           │       └── apply bonuses (5 types)
    │                           ├── update episode state
    │                           └── return (next_obs, reward, done)
    │
    └── GET  /state ─────► EmailTriageEnv.state()
                                └── EpisodeState snapshot
                                    (step, score, history, remaining)
```

---

## 📊 Grading Dimensions

### Category Grading

Binary — correct or incorrect. Worth 25–50% of score depending on task difficulty.

### Priority Grading

Partial credit system based on adjacency:
```
low → medium → high → urgent
 0      1        2      3
```
Distance of 0 = full credit. Distance of 1 = half credit. Distance of 2+ = no credit.

### Route Grading

Binary — correct team or incorrect. Worth 20–25%.

### Tag Grading

F1 score between predicted and expected tag sets. Supports partial credit for partially correct tagging.

---

## 📝 License

MIT License — free for research and commercial use.

---

## 🙏 Built With

- **FastAPI** — Web framework
- **Pydantic v2** — Type validation
- **OpenAI GPT-4o-mini** — Baseline agent
- **pytest** — Test suite
- **Docker** — Containerization
- **OpenEnv spec** — Environment standard
