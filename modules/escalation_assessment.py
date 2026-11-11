"""Escalation / risk assessment.

This module only MEASURES; a downstream conversational-AI system decides what
to do.  It produces one risk score (0-100), an urgency flag and a handoff
recommendation that the bot can act on.

Risk score (0-100)            Level      Meaning for the downstream bot
    0 - 19                    Low        bot can continue normally
   20 - 34                    Medium     bot continues, but monitor the case
   35 - 59                    High       hand off to a human agent
   60 - 100                   Critical   hand off to a senior human now

Handoff actions
    bot_continue            risk < 20
    bot_with_monitoring     20 <= risk < 35
    human_handoff           risk >= 35
    immediate_human_handoff risk >= 35 AND urgent, or a legal / fraud / safety
                            signal, or risk >= 60
"""
import re

from modules.context_manager import detect_resolution_status
from modules.emotion_analysis import EMOTION_SEVERITY
from modules.text_utils import normalize_text

HANDOFF_THRESHOLD = 35      # >= this: a human should take over
CRITICAL_THRESHOLD = 60
MEDIUM_THRESHOLD = 20

# points contributed by the customer's emotion (latest / peak)
_EMOTION_POINTS = {"anger": 30, "frustration": 20, "anxiety": 12, "sadness": 10, "confusion": 5}

_UNRESOLVED_PHRASES = [
    r"still (?:waiting|no\b|nothing|not\b|haven't|have not|hasn't|has not|pending|unresolved|broken|stuck|missing|charged)",
    r"not (?:yet )?(?:resolved|fixed|received|arrived|delivered|credited|refunded|working)",
    r"(?:haven't|have not|hasn't|has not|didn't|did not|never) (?:received|got|gotten|arrived|heard)",
    r"no (?:response|reply|answer|update|updates|resolution|progress)",
    r"(?:nobody|no ?one)\b.{0,20}\b(?:help\w*|respon\w+|repl(?:y|ies|ied|ying)|answer\w*|call\w*|contact\w*)",
    r"problem persists|issue persists|same (?:problem|issue)|remains unresolved",
    r"(?:isn't|is not|doesn't|does not|won't|will not|can't|cannot) (?:work|working|load|open|go through)",
    r"yet to (?:receive|get|hear)|not heard back",
    r"(?:it(?:'s| has| is)? been|been waiting(?: for)?) (?:over |almost |nearly |more than |about )?(?:\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|several|many|few) (?:days?|weeks?|months?)",
    r"(?:late|delayed|overdue)\b",
]
_URGENCY_PHRASES = [
    r"urgent\w*", r"immediate\w*", r"as soon as possible", r"\basap\b", r"right away", r"right now",
    r"emergency", r"without delay", r"need (?:it|this) (?:now|today|tonight|tomorrow)", r"time[- ]sensitive",
    r"deadline", r"(?:by|before) (?:today|tonight|tomorrow)",
]
_REPEATED_PHRASES = [
    r"tried (?:many|several|multiple|a few|\d+) times", r"already contacted", r"multiple attempts",
    r"contacted (?:support|you|us) before",
    r"(?:second|third|fourth|fifth|\d+(?:st|nd|rd|th)) time", r"again and again|over and over",
    r"(?:called|emailed|messaged|contacted|chatted)\b.{0,20}\b(?:twice|again|several times|many times|multiple times|\d+ times)",
]
_LEGAL_PHRASES = [
    r"lawyer", r"attorney", r"legal action", r"\bsue\b", r"consumer (?:court|forum|protection)",
    r"ombudsman", r"chargeback", r"dispute (?:the |this )?charge", r"report (?:you|this|it|the company)",
    r"social media", r"bad review|one star",
]
# Money-at-risk / safety situations that justify a human even at lower emotion.
_SAFETY_PHRASES = [
    r"fraud\w*", r"scam\w*", r"unauthori[sz]ed", r"hack\w*", r"stolen|stole\b", r"identity theft",
    r"account (?:was |has been )?(?:compromised|taken over)", r"charged twice|double[- ]charged|duplicate charge",
    r"medical|hospital|injur\w+|unsafe|dangerous",
]
_CHURN_PHRASES = [
    r"never (?:buy|order|shop|use|purchase)\b", r"never\b.{0,25}\b(?:again|come back|recommend)",
    r"cancel (?:my )?(?:account|subscription|membership)", r"close my account",
    r"switch(?:ing)? to (?:another|a different)",
]
_HUMAN_REQUEST = [
    r"(?:speak|talk) (?:to|with) (?:a |an |your |the |some )?(?:manager|supervisor|senior|real person|human|someone)",
    r"\bescalate\b", r"\bsupervisor\b", r"\bmanager\b", r"human agent", r"real person", r"\bcomplaint\b",
]
_FINANCIAL_INTENTS = {"duplicate_charge", "payment_issue", "get_refund", "track_refund"}


def _any(patterns, text):
    return any(re.search(p, text) for p in patterns)


def _level(score):
    if score >= CRITICAL_THRESHOLD:
        return "Critical"
    if score >= HANDOFF_THRESHOLD:
        return "High"
    if score >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def assess_escalation(messages, emotion_results, progression, contact_stats, intent_summary):
    customer_messages = [m for m in messages if m.get("speaker", "").lower() == "customer"]
    combined = " ".join(normalize_text(m.get("message", "")).lower() for m in customer_messages)
    resolution = detect_resolution_status(messages)

    points = []   # (points, reason)

    ordered = sorted(emotion_results or [], key=lambda x: x.get("sequence_number") or 0)
    emotions = [(e.get("emotion") or "neutral").lower() for e in ordered]
    latest = emotions[-1] if emotions else "neutral"
    peak = max(emotions, key=lambda e: _EMOTION_POINTS.get(e, 0), default="neutral")

    # 1. Emotion: the strongest of the latest emotion and 80% of the earlier peak.
    latest_pts = _EMOTION_POINTS.get(latest, 0)
    peak_pts = int(_EMOTION_POINTS.get(peak, 0) * 0.8)
    if latest_pts or peak_pts:
        if latest_pts >= peak_pts:
            points.append((latest_pts, f"Customer is showing {latest}."))
        else:
            points.append((peak_pts, f"Customer showed {peak} earlier in the conversation."))
    high_intensity = sum(1 for e in ordered if e.get("intensity") == "high")
    if high_intensity:
        points.append((5, "Strongly worded negative message(s) detected."))

    # 2. Trend
    if (progression or {}).get("progression") == "Deteriorating":
        points.append((8, "Customer emotion is getting worse over the conversation."))

    # 3. Unresolved issue wording
    if _any(_UNRESOLVED_PHRASES, combined) or resolution["status"] == "Unresolved":
        points.append((15, "An unresolved customer issue is indicated."))

    # 4. Repeated attempts / repeat contacts
    if _any(_REPEATED_PHRASES, combined):
        points.append((15, "Repeated unsuccessful resolution attempts indicated."))
    contacts = (contact_stats or {}).get("customer_contacted_conversations", 0)
    if contacts >= 3:
        points.append((10, "Customer has contacted support repeatedly."))
    elif contacts >= 2:
        points.append((5, "Customer has contacted support more than once."))

    # 5. Legal / public exposure, safety and fraud
    legal = _any(_LEGAL_PHRASES, combined)
    safety = _any(_SAFETY_PHRASES, combined)
    if legal:
        points.append((35, "Customer mentioned legal, chargeback or public-escalation action."))
    if safety:
        points.append((15, "Fraud, unauthorised-charge, duplicate-charge or safety wording detected."))

    # 6. Customer asks for a person / complaint
    if _any(_HUMAN_REQUEST, combined):
        points.append((12, "Customer asked for a manager/human or raised a formal complaint."))

    # 7. Churn
    if _any(_CHURN_PHRASES, combined):
        points.append((8, "Customer signalled they may leave or stop buying."))

    # 8. Money involved
    intent = ((intent_summary or {}).get("primary_intent") or (intent_summary or {}).get("main_intent") or "").lower()
    if intent in _FINANCIAL_INTENTS and (latest_pts or peak_pts or _any(_UNRESOLVED_PHRASES, combined)):
        points.append((8, f"Money-related issue ({intent.replace('_', ' ')}) is involved."))

    # 9. Urgency
    urgent = _any(_URGENCY_PHRASES, combined) or any(e.get("urgent") for e in ordered) or latest == "urgency"
    if urgent:
        points.append((20, "Customer explicitly expressed urgency."))

    raw = min(100, sum(p for p, _ in points))
    reasons = [r for _, r in points]

    # 10. If the customer says it is sorted, the risk is much lower (not gone: a past
    #     threat or fraud report still deserves a look, hence the floor of the cap).
    if resolution["resolved"]:
        score = int(min(raw * 0.25, MEDIUM_THRESHOLD - 1))
        reasons = ["Customer's latest message indicates the issue is resolved; risk reduced."] + reasons
    else:
        score = int(raw)

    level = _level(score)

    # ---- handoff recommendation for the downstream bot ----------------------
    emergency = (score >= HANDOFF_THRESHOLD and urgent) or ((legal or safety) and score >= HANDOFF_THRESHOLD) or score >= CRITICAL_THRESHOLD
    if emergency:
        action, why = "immediate_human_handoff", "High risk combined with urgency, a legal/fraud/safety signal, or critical risk."
    elif score >= HANDOFF_THRESHOLD:
        action, why = "human_handoff", f"Risk score {score} is at or above the handoff threshold ({HANDOFF_THRESHOLD})."
    elif score >= MEDIUM_THRESHOLD:
        action, why = "bot_with_monitoring", "Moderate risk: bot may continue but the case should be monitored."
    else:
        action, why = "bot_continue", "Low risk: no human involvement indicated."

    return {
        # new, preferred fields
        "risk_score": score, "risk_level": level, "urgent": bool(urgent),
        "handoff_action": action, "handoff_reason": why,
        "handoff_required": action in ("human_handoff", "immediate_human_handoff"),
        # backward-compatible fields used by app.py / routing / summary
        "priority": level, "score": score, "escalation_score": score,
        "reasons": reasons,
        "human_review_required": level in ("High", "Critical"),
        "resolution_status": resolution["status"], "resolved": resolution["resolved"],
    }
