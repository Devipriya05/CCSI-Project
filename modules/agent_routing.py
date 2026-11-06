INTENT_DEPARTMENT_MAP = {
    "duplicate_charge": "Billing and Payments", "payment_verification": "Billing and Payments",
    "payment_verified": "Billing and Payments", "payment_issue": "Billing and Payments",
    "check_invoice": "Billing and Payments", "check_payment_methods": "Billing and Payments",
    "get_refund": "Refund and Returns", "track_refund": "Refund and Returns", "check_refund_policy": "Refund and Returns",
    "track_order": "Order and Delivery Support", "place_order": "Order and Delivery Support",
    "cancel_order": "Order and Delivery Support", "change_order": "Order and Delivery Support",
    "delivery_period": "Order and Delivery Support", "recover_password": "Account Support",
    "create_account": "Account Support", "delete_account": "Account Support", "edit_account": "Account Support",
    "technical_issue": "Technical Support", "contact_customer_service": "General Customer Service",
    "issue_resolved": "General Customer Service"
}


def recommend_agent(intent_summary, escalation_result):
    intent = (intent_summary.get("primary_intent") or intent_summary.get("main_intent") or
              intent_summary.get("latest_intent") or "").lower()
    resolved = bool(escalation_result.get("resolved", False))
    department = INTENT_DEPARTMENT_MAP.get(intent, "General Customer Service")
    priority = escalation_result.get("priority", "Low")
    if resolved:
        status = "No assignment recommended - case appears resolved"
        agent_type = "No immediate agent assignment"
    else:
        status = "Recommendation - Awaiting Agent Assignment"
        agent_type = "Senior Support Agent" if priority == "Critical" else "Experienced Support Agent" if priority == "High" else "Standard Support Agent"
    return {"predicted_intent": intent, "recommended_department": department,
            "recommended_agent_type": agent_type, "priority": priority, "routing_status": status}
