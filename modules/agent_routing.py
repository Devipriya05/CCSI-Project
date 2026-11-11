INTENT_DEPARTMENT_MAP = {
    # Billing and payments
    "duplicate_charge": "Billing and Payments", "payment_verification": "Billing and Payments",
    "payment_verified": "Billing and Payments", "payment_issue": "Billing and Payments",
    "check_invoice": "Billing and Payments", "get_invoice": "Billing and Payments",
    "check_payment_methods": "Billing and Payments", "check_cancellation_fee": "Billing and Payments",
    # Refunds and returns
    "get_refund": "Refund and Returns", "track_refund": "Refund and Returns",
    "check_refund_policy": "Refund and Returns",
    # Orders and delivery
    "track_order": "Order and Delivery Support", "place_order": "Order and Delivery Support",
    "cancel_order": "Order and Delivery Support", "change_order": "Order and Delivery Support",
    "delivery_period": "Order and Delivery Support", "delivery_options": "Order and Delivery Support",
    "change_shipping_address": "Order and Delivery Support", "set_up_shipping_address": "Order and Delivery Support",
    # Accounts
    "recover_password": "Account Support", "create_account": "Account Support",
    "delete_account": "Account Support", "edit_account": "Account Support",
    "switch_account": "Account Support", "registration_problems": "Account Support",
    "newsletter_subscription": "Account Support",
    # Technical
    "technical_issue": "Technical Support",
    # General
    "contact_customer_service": "General Customer Service", "contact_human_agent": "General Customer Service",
    "complaint": "Complaints and Escalations", "review": "General Customer Service",
    "issue_resolved": "General Customer Service", "smalltalk": "General Customer Service",
    "unclear": "General Customer Service",
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
            "recommended_agent_type": agent_type, "priority": priority, "routing_status": status,
            # Aliases: the dashboard (app.py) reads these two keys.
            "recommended_agent_level": agent_type, "routing_priority": priority}
