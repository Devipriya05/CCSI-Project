from modules.context_manager import detect_resolution_status


def generate_case_summary(customer_id, messages, intent_summary, emotion_results, progression,
                          contact_stats, claim_results, escalation_result, routing_result):
    customer = [m for m in messages if m.get("speaker", "").lower() == "customer"]
    first_issue = customer[0].get("message", "No customer message available.") if customer else "No customer message available."
    latest_message = customer[-1].get("message", "No customer message available.") if customer else "No customer message available."
    latest_emotion = emotion_results[-1].get("emotion", "Unknown") if emotion_results else "Unknown"
    inconsistencies = claim_results.get("total_inconsistencies", 0)
    resolution = detect_resolution_status(messages)
    resolved = resolution["resolved"]
    priority = escalation_result.get("priority", "Low")
    if resolved:
        next_action = "Mark the case as resolved or close it according to the support workflow; no further customer follow-up is currently indicated."
    elif priority == "Critical":
        next_action = "Request immediate senior-agent review and prioritize the unresolved customer issue."
    elif priority == "High":
        next_action = "Assign to an experienced support agent for prompt investigation."
    elif inconsistencies:
        next_action = "Request clarification about the potentially inconsistent statements; human verification is required."
    elif priority == "Medium":
        next_action = "Investigate the issue and provide the customer with an appropriate update."
    else:
        next_action = "Continue with standard customer-support procedures and confirm the outcome with the customer."
    if inconsistencies and not resolved:
        next_action += " Human verification of the potential inconsistency is required."
    primary_intent = intent_summary.get("primary_intent") or intent_summary.get("main_intent", "Unknown")
    return {
        "customer_id": customer_id, "customer_issue": first_issue, "latest_customer_message": latest_message,
        "main_intent": primary_intent, "primary_intent": primary_intent,
        "latest_intent": intent_summary.get("latest_intent", "Unknown"),
        "latest_emotion": latest_emotion, "emotion_progression": progression.get("progression", "Unknown"),
        "resolution_status": resolution["status"], "resolution_evidence": resolution["evidence"],
        "total_customer_contacts": contact_stats.get("customer_contacted_conversations", 0),
        "potential_inconsistencies": inconsistencies, "claim_evidence": claim_results.get("inconsistencies", []),
        "escalation_priority": priority, "escalation_score": escalation_result.get("score", 0),
        "escalation_reasons": escalation_result.get("reasons", []),
        "recommended_department": routing_result.get("recommended_department", "General Customer Service"),
        "recommended_agent_type": routing_result.get("recommended_agent_type", "Standard Support Agent"),
        "recommended_next_action": next_action,
        "human_review_required": bool(escalation_result.get("human_review_required", False) or (inconsistencies > 0 and not resolved))
    }
