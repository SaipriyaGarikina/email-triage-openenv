"""
scripts/baseline.py  (v2 — Fixed)
-----------------------------------
Standalone baseline using GPT-4o-mini. Deterministic (temperature=0).
Saves results to baseline_results.json.

FIXES APPLIED:
  [Fix #1] Unpacks 4-tuple from env.step() per OpenEnv spec
  [Fix #7] Results file now reflects the actual model used (gpt-4o-mini)

Usage:
  export OPENAI_API_KEY="sk-..."
  python scripts/baseline.py
  python scripts/baseline.py --task easy_triage
  python scripts/baseline.py --quiet
"""

from __future__ import annotations

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from app.env import EmailTriageEnv
from app.models import Action, Category, Priority, RouteTarget
from app.graders import grade_full_episode
from app.tasks import list_tasks, TASK_REGISTRY

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE  = 0
MAX_TOKENS   = 400

SYSTEM_PROMPT = """You are an expert email triage specialist working at a SaaS customer operations center.

Your job is to read incoming emails and decide:
1. category  : spam | support | billing | sales | hr | legal | general
2. priority  : low | medium | high | urgent
3. route_to  : sales_team | tech_team | billing_team | hr_team | legal_team | trash | inbox
4. tags      : list of 2-4 relevant lowercase keyword tags
5. notes     : brief reasoning (1-2 sentences)

=== STRICT CLASSIFICATION RULES ===

SPAM:
- Lottery wins, prize claims, phishing, unsolicited mass email → spam + trash + low

SUPPORT:
- Technical issues, login problems, app bugs, broken features → support + tech_team
- Priority: low (informational), medium (inconvenient), high (blocking work), urgent (data loss/outage)

BILLING:
- Invoices, payments, refunds, subscription issues, failed charges → billing + billing_team
- Priority based on urgency and customer tone

SALES:
- Pricing questions, upgrade inquiries, demos, enterprise deals, partnerships → sales + sales_team
- Enterprise deals (large volume, procurement deadlines) → HIGH priority

HR:
- Job applications, employee complaints, harassment reports, workplace issues → hr + hr_team
- Harassment/complaints → HIGH priority
- Job applications → LOW priority

LEGAL:
- Contracts, legal disputes, data breaches, GDPR, compliance, regulatory → legal + legal_team
- Data Processing Agreements (DPA), GDPR Article 28/33 requests → legal + legal_team
- A prospect asking for a DPA BEFORE a commercial deal → legal (not sales), HIGH
- Data breaches → URGENT
- Contract disputes, DPA requests → HIGH

SPAM (ADVANCED):
- Emails from lookalike domains (e.g. ourcompany-systems.xyz instead of ourcompany.com) → spam
- Internal IT notices that ask for credential re-verification via external links → spam + trash
- Any link to .xyz / unusual TLD asking for login → spam + trash + low

GENERAL:
- Unsubscribe requests, newsletters, vague emails → general + inbox + low

=== PRIORITY RULES ===
urgent : system outage, data breach, legal deadline <24h, active security threat
high   : angry customer, revenue at risk, time-sensitive escalation, harassment complaint
medium : standard support, pricing inquiry, general sales interest
low    : newsletter, unsubscribe, informational, job application, no urgency stated

Respond with ONLY valid JSON. No preamble, no markdown fences.
"""


def run_task(client: OpenAI, task_id: str, verbose: bool = True) -> dict:
    """Run GPT-4o-mini on a single task and return graded results."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running task: {task_id}")
        print("="*60)

    env = EmailTriageEnv()
    obs = env.reset(task_id)
    done = False
    all_actions = []

    while not done:
        if verbose:
            print(f"\n[Step {obs.step + 1}/{obs.total_steps}] Email: {obs.email_id}")
            print(f"  Subject: {obs.subject}")

        user_message = (
            f"Email ID: {obs.email_id}\n"
            f"From: {obs.sender}\n"
            f"Subject: {obs.subject}\n"
            f"Body:\n{obs.body}\n\n"
            f"Classify this email. Respond ONLY with valid JSON."
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        raw = response.choices[0].message.content.strip()
        if verbose:
            print(f"  Response: {raw}")

        try:
            parsed = json.loads(raw)
            action = Action(
                email_id=obs.email_id,
                category=Category(parsed.get("category", "general")),
                priority=Priority(parsed.get("priority", "medium")),
                route_to=RouteTarget(parsed.get("route_to", "inbox")),
                tags=parsed.get("tags", []),
                notes=parsed.get("notes", ""),
            )
        except Exception as err:
            if verbose:
                print(f"  ⚠️ Parse error: {err}. Using defaults.")
            action = Action(
                email_id=obs.email_id,
                category=Category.GENERAL,
                priority=Priority.MEDIUM,
                route_to=RouteTarget.INBOX,
                tags=[],
                notes=f"Parse error: {err}",
            )

        all_actions.append(action)
        # FIX #1: unpack 4-tuple per OpenEnv spec
        next_obs, reward, done, _ = env.step(action)

        if verbose:
            print(f"  Score     : {reward.step_score:.4f}")
            print(f"  Cumulative: {reward.cumulative_score:.4f}")
            if reward.penalty > 0:
                print(f"  Penalty   : -{reward.penalty:.4f}")
            print(f"  Feedback  : {reward.feedback}")

        obs = next_obs

    result = grade_full_episode(task_id, all_actions)
    result["model"] = OPENAI_MODEL  # FIX #7: always reflects the actual model used

    if verbose:
        print(f"\n📊 RESULT — {task_id}")
        print(f"   Score     : {result['total_score']:.4f}")
        print(f"   Threshold : {TASK_REGISTRY[task_id]['info'].pass_threshold}")
        print(f"   Status    : {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
        print(f"\n   Per-email breakdown:")
        for es in result["per_email_scores"]:
            bd = es.get("breakdown", {})
            bd_str = "  ".join(f"{k}={v:.2f}" for k, v in bd.items()) if bd else "n/a"
            print(f"   [{es['email_id']}] {es['step_score']:.4f}  ({bd_str})")
            print(f"     → {es['feedback']}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description=f"Run Email Triage baseline using {OPENAI_MODEL}."
    )
    parser.add_argument(
        "--task",
        choices=["easy_triage", "medium_triage", "hard_triage"],
        default=None,
        help="Run a single task. Default: run all 3.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not set.")
        print("   Run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    client  = OpenAI(api_key=api_key)
    verbose = not args.quiet

    task_ids    = [args.task] if args.task else ["easy_triage", "medium_triage", "hard_triage"]
    all_results = []

    for task_id in task_ids:
        result = run_task(client, task_id, verbose=verbose)
        all_results.append(result)

    print(f"\n{'='*65}")
    print(f"BASELINE SUMMARY  |  Model: {OPENAI_MODEL}")
    print(f"{'='*65}")
    print(f"{'Task':<22} {'Score':>8} {'Threshold':>10} {'Status':>8}")
    print("-"*65)
    for r in all_results:
        threshold = TASK_REGISTRY[r["task_id"]]["info"].pass_threshold
        status    = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"{r['task_id']:<22} {r['total_score']:>8.4f} {threshold:>10} {status:>8}")
    print("="*65)

    avg = sum(r["total_score"] for r in all_results) / len(all_results)
    print(f"\nOverall Average Score: {avg:.4f}")

    # FIX #7: save with correct model name and accurate passed flags
    output = {
        "model": OPENAI_MODEL,
        "temperature": TEMPERATURE,
        "results": all_results,
        "summary": {r["task_id"]: r["total_score"] for r in all_results},
        "overall_average": round(avg, 4),
        "all_passed": all(r["passed"] for r in all_results),
    }
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baseline_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
