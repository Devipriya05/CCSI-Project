
from modules.context_manager import detect_resolution_status


def assess_escalation(
    messages,
    emotion_results,
    progression,
    contact_stats,
    intent_summary
):
    """
    Context-aware customer support escalation assessment.

    Considers:
    - Resolution status
    - Latest customer emotion
    - Emotional progression
    - Unresolved issue indicators
    - Customer urgency
    - Repeated customer contacts
    - Complaint/refund intent
    """

    customer_messages = [
        m for m in messages
        if m.get("speaker", "").lower() == "customer"
    ]

    combined = " ".join(
        m.get("message", "").lower()
        for m in customer_messages
    )

    resolution = detect_resolution_status(messages)

    score = 0
    reasons = []

    # Get the latest customer emotion chronologically
    ordered_emotions = sorted(
        emotion_results or [],
        key=lambda x: x.get("sequence_number", 0)
    )

    latest_emotion = (
        ordered_emotions[-1].get("emotion", "neutral").lower()
        if ordered_emotions
        else "neutral"
    )

    # 1. Repeated customer contacts
    contacts = (
        contact_stats or {}
    ).get("customer_contacted_conversations", 0)

    if contacts >= 3:
        score += 2
        reasons.append("Customer has contacted support repeatedly.")
    elif contacts >= 2:
        score += 1
        reasons.append("Customer has contacted support more than once.")

    # 2. Latest customer emotion
    if latest_emotion == "anger":
        score += 3
        reasons.append("Customer is showing anger.")

    elif latest_emotion == "frustration":
        score += 2
        reasons.append("Customer is showing frustration.")

    elif latest_emotion in ("sadness", "confusion"):
        score += 1
        reasons.append(
            f"Latest detected emotion is {latest_emotion}."
        )

    # 3. Emotional progression
    progression_status = (
        progression or {}
    ).get("progression", "Stable")

    if progression_status == "Deteriorating":
        score += 1
        reasons.append("Customer emotion is deteriorating.")

    elif progression_status == "Deteriorated then improved":
        score += 1
        reasons.append(
            "Customer experienced emotional deterioration earlier."
        )

    # 4. Unresolved issue indicators
    unresolved_phrases = [
        "still waiting",
        "not resolved",
        "not fixed",
        "not received",
        "haven't received",
        "have not received",
        "hasn't arrived",
        "has not arrived",
        "no response",
        "problem persists",
        "still not working",
        "refund is pending",
        "refund has not arrived"
    ]

    if any(phrase in combined for phrase in unresolved_phrases):
        score += 2
        reasons.append("An unresolved customer issue is indicated.")

    # 5. Customer urgency
    urgency_phrases = [
        "urgent",
        "immediately",
        "as soon as possible",
        "right away",
        "emergency",
        "without delay",
        "need it now"
    ]

    if any(phrase in combined for phrase in urgency_phrases):
        score += 1
        reasons.append("Customer explicitly expressed urgency.")

    # 6. Refund or complaint-related intent
    intent = (
        (intent_summary or {}).get("primary_intent")
        or (intent_summary or {}).get("main_intent")
        or ""
    ).lower()

    if not resolution["resolved"]:
        if (
            "refund" in intent
            or "complaint" in combined
            or "supervisor" in combined
            or "manager" in combined
        ):
            score += 1
            reasons.append(
                "Unresolved refund or complaint-related request detected."
            )

    # 7. Repeated unsuccessful attempts
    repeated_phrases = [
        "tried many times",
        "already contacted",
        "multiple attempts",
        "still not fixed",
        "problem persists",
        "contacted support before"
    ]

    if any(phrase in combined for phrase in repeated_phrases):
        score += 2
        reasons.append(
            "Repeated unsuccessful resolution attempts indicated."
        )

    # 8. Resolution-aware adjustment
    if resolution["resolved"]:
        score = 0
        reasons = [
            "Customer's issue is confirmed as resolved; "
            "no escalation is currently indicated."
        ]

    # 9. Priority classification
    if score >= 11:
        priority = "Critical"
    elif score >= 8:
        priority = "High"
    elif score >= 3:
        priority = "Medium"
    else:
        priority = "Low"

    return {
        "priority": priority,
        "score": score,
        "reasons": reasons,
        "human_review_required": priority in ("High", "Critical"),
        "resolution_status": resolution["status"],
        "resolved": resolution["resolved"],
        "escalation_score": score
    }
