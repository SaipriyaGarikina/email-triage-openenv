"""
tasks.py  (v3 — Enhanced Hard Task)
-------------------------------------
Defines all three tasks (easy, medium, hard) with their email datasets
and ground-truth labels used by the graders.

ENHANCEMENTS in v3:
  Hard task upgraded from 7 → 10 emails with 3 new adversarial cases:
    hard_008 — Billing disguised as support (angry but polite tone)
    hard_009 — Urgent sales opportunity with legal undertone
    hard_010 — Internal-looking phishing with plausible domain

  THREAD_CONTEXT entries added for hard_008 and hard_010.

Hard task openenv.yaml email_count and max_steps updated to match.
"""

from __future__ import annotations

from app.models import (
    Category, Priority, RouteTarget, Difficulty, Email, TaskInfo
)

# ---------------------------------------------------------------------------
# TASK 1 — EASY: 3 simple, unambiguous emails
# ---------------------------------------------------------------------------

EASY_EMAILS: list[Email] = [
    Email(
        email_id="easy_001",
        subject="YOU WON $1,000,000 CLICK NOW!!!",
        body=(
            "Congratulations! You have been selected as our lucky winner. "
            "Click the link below to claim your prize immediately. "
            "This is a limited-time offer. Act NOW before it expires!"
        ),
        sender="prize_winner@totally-legit-lottery.ru",
        timestamp="2024-06-01T09:00:00Z",
    ),
    Email(
        email_id="easy_002",
        subject="My account login is not working",
        body=(
            "Hi Support Team, I have been trying to log into my account "
            "since yesterday but it keeps saying 'invalid password'. "
            "I tried resetting but the reset email never arrived. "
            "Can you please help me access my account? My username is john.doe@gmail.com."
        ),
        sender="john.doe@gmail.com",
        timestamp="2024-06-01T10:30:00Z",
    ),
    Email(
        email_id="easy_003",
        subject="Invoice #INV-4821 – Payment Due",
        body=(
            "Dear Customer, please find attached invoice #INV-4821 "
            "for the amount of $299.00 due on 2024-06-15. "
            "Kindly make the payment via bank transfer or credit card. "
            "If you have already paid, please ignore this reminder."
        ),
        sender="accounts@vendor-corp.com",
        timestamp="2024-06-01T11:00:00Z",
    ),
]

EASY_GROUND_TRUTH: dict[str, dict] = {
    "easy_001": {
        "category": Category.SPAM,
        "priority": Priority.LOW,
        "route_to": RouteTarget.TRASH,
        "tags": ["spam", "phishing"],
    },
    "easy_002": {
        "category": Category.SUPPORT,
        "priority": Priority.HIGH,
        "route_to": RouteTarget.TECH_TEAM,
        "tags": ["login", "account_access"],
    },
    "easy_003": {
        "category": Category.BILLING,
        "priority": Priority.MEDIUM,
        "route_to": RouteTarget.BILLING_TEAM,
        "tags": ["invoice", "payment"],
    },
}

EASY_TASK_INFO = TaskInfo(
    task_id="easy_triage",
    name="Easy Email Triage",
    difficulty=Difficulty.EASY,
    description=(
        "Classify 3 simple emails into correct categories (spam, support, billing). "
        "Emails are clearly worded with no ambiguity."
    ),
    email_count=3,
    max_steps=10,
    pass_threshold=0.80,
)

# ---------------------------------------------------------------------------
# TASK 2 — MEDIUM: 5 emails with priority assignment
# ---------------------------------------------------------------------------

MEDIUM_EMAILS: list[Email] = [
    Email(
        email_id="med_001",
        subject="Urgent: Production server is down!",
        body=(
            "ALERT: Our production server has been unreachable for the past 30 minutes. "
            "All customer transactions are failing. Revenue loss is approximately $5000/minute. "
            "Engineering team is investigating but needs immediate escalation. "
            "Please respond ASAP. CEO is on the phone."
        ),
        sender="devops@ourcompany.com",
        timestamp="2024-06-02T02:15:00Z",
    ),
    Email(
        email_id="med_002",
        subject="Question about your premium plan",
        body=(
            "Hello, I am considering upgrading from the basic plan to premium. "
            "Could you tell me what extra features I would get? "
            "Also, is there a discount for annual billing? "
            "No rush, just exploring options. Thanks!"
        ),
        sender="potential.buyer@hotmail.com",
        timestamp="2024-06-02T09:00:00Z",
    ),
    Email(
        email_id="med_003",
        subject="Re: My refund request from last week",
        body=(
            "I sent a refund request on May 26th for order #ORD-9921 but have not heard back. "
            "It has been 7 days and my bank account still shows the charge. "
            "I am getting frustrated. Can someone please look into this? "
            "I may have to escalate to my bank if this is not resolved soon."
        ),
        sender="angry.customer@yahoo.com",
        timestamp="2024-06-02T10:45:00Z",
    ),
    Email(
        email_id="med_004",
        subject="Newsletter unsubscribe request",
        body=(
            "Please remove me from your mailing list. "
            "I have clicked unsubscribe multiple times but keep receiving emails. "
            "My email is newsletter_victim@gmail.com."
        ),
        sender="newsletter_victim@gmail.com",
        timestamp="2024-06-02T11:00:00Z",
    ),
    Email(
        email_id="med_005",
        subject="Partnership opportunity – AI integration",
        body=(
            "Hi, I am the Business Development Manager at TechFlow Inc. "
            "We are interested in integrating your API into our platform. "
            "We currently have 50,000 active users and are looking for a white-label solution. "
            "Could we schedule a call this week to discuss terms? "
            "This could be a significant revenue opportunity for both companies."
        ),
        sender="biz.dev@techflow.io",
        timestamp="2024-06-02T14:30:00Z",
    ),
]

MEDIUM_GROUND_TRUTH: dict[str, dict] = {
    "med_001": {
        "category": Category.SUPPORT,
        "priority": Priority.URGENT,
        "route_to": RouteTarget.TECH_TEAM,
        "tags": ["outage", "production", "escalation"],
    },
    "med_002": {
        "category": Category.SALES,
        "priority": Priority.MEDIUM,
        "route_to": RouteTarget.SALES_TEAM,
        "tags": ["upgrade", "pricing"],
    },
    "med_003": {
        "category": Category.BILLING,
        "priority": Priority.HIGH,
        "route_to": RouteTarget.BILLING_TEAM,
        "tags": ["refund", "delayed_response", "angry_customer"],
    },
    "med_004": {
        "category": Category.GENERAL,
        "priority": Priority.LOW,
        "route_to": RouteTarget.INBOX,
        "tags": ["unsubscribe", "mailing_list"],
    },
    "med_005": {
        "category": Category.SALES,
        "priority": Priority.HIGH,
        "route_to": RouteTarget.SALES_TEAM,
        "tags": ["partnership", "api", "b2b"],
    },
}

MEDIUM_TASK_INFO = TaskInfo(
    task_id="medium_triage",
    name="Medium Email Triage + Priority",
    difficulty=Difficulty.MEDIUM,
    description=(
        "Classify 5 emails AND assign priority (low, medium, high, urgent). "
        "Some emails are ambiguous and require reasoning."
    ),
    email_count=5,
    max_steps=15,
    pass_threshold=0.75,
)

# ---------------------------------------------------------------------------
# TASK 3 — HARD: 10 emails requiring full reasoning + adversarial traps
#
# Original 7 emails retained unchanged.
# 3 new adversarial emails added:
#   hard_008 — billing issue written in a polite, support-sounding tone
#   hard_009 — sales email that contains legal urgency signals
#   hard_010 — convincing internal-sounding phishing from plausible domain
# ---------------------------------------------------------------------------

HARD_EMAILS: list[Email] = [
    # ---- Original 7 (unchanged) ----
    Email(
        email_id="hard_001",
        subject="Data breach notification – immediate action required",
        body=(
            "We have detected unusual login activity on your platform from IP 192.168.1.100. "
            "Over 2,000 user records may have been accessed without authorization. "
            "Under GDPR Article 33, you are required to notify the supervisory authority within 72 hours. "
            "Our legal team needs to be involved immediately. "
            "Please do not communicate this externally until our lawyers have reviewed."
        ),
        sender="security.alert@internalmonitoring.com",
        timestamp="2024-06-03T03:00:00Z",
    ),
    Email(
        email_id="hard_002",
        subject="Employee complaint – workplace harassment",
        body=(
            "I need to report a serious issue. My manager, Mr. Stevens, has been "
            "making inappropriate comments about my appearance in team meetings. "
            "I have documented three incidents in the past two weeks. "
            "I am uncomfortable and would like this handled discreetly. "
            "Please let me know the next steps for a formal complaint."
        ),
        sender="employee.confidential@ourcompany.com",
        timestamp="2024-06-03T08:00:00Z",
    ),
    Email(
        email_id="hard_003",
        subject="Your subscription renewal failed",
        body=(
            "Hi, my subscription auto-renewal failed yesterday because my card expired. "
            "I have updated my payment method in the dashboard but the system still shows "
            "my account as suspended. I need access restored urgently because I have a "
            "client presentation in 3 hours and I need to download my project files."
        ),
        sender="freelancer.client@designstudio.com",
        timestamp="2024-06-03T09:30:00Z",
    ),
    Email(
        email_id="hard_004",
        subject="Re: Contract renewal terms – Section 4.2 dispute",
        body=(
            "Further to our last call, I want to formally object to the new liability clause "
            "under Section 4.2 of the renewal contract. Our legal counsel believes this clause "
            "exposes us to unlimited indemnification which is unacceptable. "
            "We propose either removing Section 4.2 entirely or capping liability at $50,000. "
            "Please have your legal team review and respond before the contract expires on June 10."
        ),
        sender="contracts@enterprise-client.com",
        timestamp="2024-06-03T10:00:00Z",
    ),
    Email(
        email_id="hard_005",
        subject="Can you do a quick custom report for me?",
        body=(
            "Hey, hope you're doing well! My name is Sarah and I'm a long-time customer. "
            "I was wondering if your team could pull a custom analytics report for my account "
            "showing all transactions from Q1. I know this might not be a standard feature "
            "but I'd really appreciate it. Happy to pay extra if needed. "
            "Not urgent, just whenever you have bandwidth!"
        ),
        sender="sarah.k@smallbiz.net",
        timestamp="2024-06-03T11:00:00Z",
    ),
    Email(
        email_id="hard_006",
        subject="Job application – Senior Data Analyst position",
        body=(
            "Dear Hiring Manager, please find attached my resume for the Senior Data Analyst "
            "position posted on LinkedIn (Job ID: DA-2024-011). "
            "I have 6 years of experience in data analytics, SQL, and Python. "
            "I am available for interviews starting next week. "
            "Looking forward to hearing from you."
        ),
        sender="applicant.resume@gmail.com",
        timestamp="2024-06-03T12:00:00Z",
    ),
    Email(
        email_id="hard_007",
        subject="Bulk discount inquiry for 500 licenses",
        body=(
            "Hello Sales Team, we are a mid-sized enterprise with 500 employees and "
            "we are evaluating your platform as our company-wide solution. "
            "Could you provide a custom quote for 500 annual licenses? "
            "We also need SSO integration, a dedicated account manager, and SLA guarantees. "
            "Our procurement deadline is June 20. Decision maker is our VP of Technology."
        ),
        sender="procurement@bigenterprise.co",
        timestamp="2024-06-03T14:00:00Z",
    ),

    # ---- NEW: Adversarial email #1 ----
    # TRAP: polite, support-sounding tone — but it's BILLING + HIGH
    # Agent failure mode: classify as support/tech_team because customer mentions "dashboard bug"
    Email(
        email_id="hard_008",
        subject="Small issue with my dashboard – might be a display bug?",
        body=(
            "Hi there! Hope you're having a great day. I noticed something a bit odd — "
            "my dashboard is showing that I was charged $149 twice this month, "
            "on June 1st and again on June 3rd. I'm sure it's just a display glitch! "
            "Could someone check my billing history? My account is under lisa.m@creative.io. "
            "If it really is a double charge, I'd love a refund when convenient. "
            "No big rush, but wanted to flag it just in case. Thanks so much!"
        ),
        sender="lisa.m@creative.io",
        timestamp="2024-06-03T15:30:00Z",
    ),

    # ---- NEW: Adversarial email #2 ----
    # TRAP: looks like a sales opportunity — but contains a regulatory compliance deadline
    # that makes it LEGAL + HIGH, not SALES + HIGH
    # Agent failure mode: route to sales_team, miss the compliance angle
    Email(
        email_id="hard_009",
        subject="Interested in your enterprise plan – but need DPA first",
        body=(
            "Hello, we're a fintech company regulated under FCA guidelines and we're very "
            "interested in signing up for your Enterprise plan (500+ seats). "
            "However, before we can proceed, our compliance officer requires a signed "
            "Data Processing Agreement (DPA) under GDPR Article 28. "
            "We have a board review on June 12th and need the DPA countersigned before then. "
            "Could your legal team send over your standard DPA template? "
            "Once that's done, we're ready to move forward with the commercial discussion."
        ),
        sender="compliance@fintechco.io",
        timestamp="2024-06-03T16:00:00Z",
    ),

    # ---- NEW: Adversarial email #3 ----
    # TRAP: convincing internal-looking email from a domain that looks legitimate
    # but is NOT — it's sophisticated phishing. Subject sounds like IT policy.
    # Agent failure mode: classify as support/general, route to tech_team or inbox
    # Correct answer: SPAM + TRASH + LOW (phishing, not internal IT)
    Email(
        email_id="hard_010",
        subject="Action required: Mandatory security policy update – complete by EOD",
        body=(
            "Dear team member, as part of our ongoing SOC 2 compliance review, "
            "all employees must re-verify their credentials via our updated security portal. "
            "Please click the link below and log in with your corporate credentials within 24 hours "
            "or your account access will be suspended. "
            "This is a company-wide mandatory action — no exceptions. "
            "Link: https://ourcompany-security-portal.xyz/verify?token=a8f3k2 "
            "— IT Security Team"
        ),
        sender="it-security@ourcompany-systems.xyz",
        timestamp="2024-06-03T17:00:00Z",
    ),
]

HARD_GROUND_TRUTH: dict[str, dict] = {
    # ---- Original 7 ----
    "hard_001": {
        "category": Category.LEGAL,
        "priority": Priority.URGENT,
        "route_to": RouteTarget.LEGAL_TEAM,
        "tags": ["data_breach", "gdpr", "security", "urgent_legal"],
    },
    "hard_002": {
        "category": Category.HR,
        "priority": Priority.HIGH,
        "route_to": RouteTarget.HR_TEAM,
        "tags": ["harassment", "complaint", "confidential"],
    },
    "hard_003": {
        "category": Category.BILLING,
        "priority": Priority.URGENT,
        "route_to": RouteTarget.BILLING_TEAM,
        "tags": ["subscription", "payment_failed", "account_suspended", "time_sensitive"],
    },
    "hard_004": {
        "category": Category.LEGAL,
        "priority": Priority.HIGH,
        "route_to": RouteTarget.LEGAL_TEAM,
        "tags": ["contract", "dispute", "liability", "renewal"],
    },
    "hard_005": {
        "category": Category.SUPPORT,
        "priority": Priority.LOW,
        "route_to": RouteTarget.TECH_TEAM,
        "tags": ["custom_report", "analytics", "feature_request"],
    },
    "hard_006": {
        "category": Category.HR,
        "priority": Priority.LOW,
        "route_to": RouteTarget.HR_TEAM,
        "tags": ["job_application", "hiring", "resume"],
    },
    "hard_007": {
        "category": Category.SALES,
        "priority": Priority.HIGH,
        "route_to": RouteTarget.SALES_TEAM,
        "tags": ["enterprise", "bulk_license", "sso", "procurement"],
    },

    # ---- New adversarial 3 ----
    "hard_008": {
        # Polite tone hides a real double-charge billing dispute
        "category": Category.BILLING,
        "priority": Priority.HIGH,
        "route_to": RouteTarget.BILLING_TEAM,
        "tags": ["double_charge", "refund", "billing_error"],
    },
    "hard_009": {
        # DPA + GDPR compliance need makes this legal, not sales
        # Route to legal first; sales can follow up after DPA is signed
        "category": Category.LEGAL,
        "priority": Priority.HIGH,
        "route_to": RouteTarget.LEGAL_TEAM,
        "tags": ["dpa", "gdpr", "compliance", "enterprise_prospect"],
    },
    "hard_010": {
        # Convincing phishing — external domain masquerading as internal IT
        "category": Category.SPAM,
        "priority": Priority.LOW,
        "route_to": RouteTarget.TRASH,
        "tags": ["phishing", "credential_theft", "fake_it"],
    },
}

HARD_TASK_INFO = TaskInfo(
    task_id="hard_triage",
    name="Hard Email Triage + Route + Tag",
    difficulty=Difficulty.HARD,
    description=(
        "Classify 10 complex and adversarial emails, assign priority, tag with topic labels, "
        "and route to the correct team. Includes deceptive cases: billing disguised as support, "
        "sales emails with legal blockers, and sophisticated phishing. "
        "Full contextual reasoning required. Designed to challenge frontier models."
    ),
    email_count=10,
    max_steps=25,
    pass_threshold=0.65,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, dict] = {
    "easy_triage": {
        "info": EASY_TASK_INFO,
        "emails": EASY_EMAILS,
        "ground_truth": EASY_GROUND_TRUTH,
    },
    "medium_triage": {
        "info": MEDIUM_TASK_INFO,
        "emails": MEDIUM_EMAILS,
        "ground_truth": MEDIUM_GROUND_TRUTH,
    },
    "hard_triage": {
        "info": HARD_TASK_INFO,
        "emails": HARD_EMAILS,
        "ground_truth": HARD_GROUND_TRUTH,
    },
}


def get_task(task_id: str) -> dict:
    """Return the full task dict for a given task_id."""
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Valid options: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]


def list_tasks() -> list[TaskInfo]:
    """Return info about all available tasks."""
    return [v["info"] for v in TASK_REGISTRY.values()]
