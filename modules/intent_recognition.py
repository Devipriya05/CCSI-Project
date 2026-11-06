import os
import re
import joblib

MODEL_PATH = os.path.join("models", "intent_classifier.joblib")

# High-precision patterns for common support intents that are often poorly represented
# in small TF-IDF training sets. Rule matches override the model for that message only.
RULE_INTENTS = [
    ("duplicate_charge", [r"\bcharged twice\b", r"\bdouble[- ]charged\b", r"\bduplicate charge\b", r"\bcharged more than once\b"]),
    ("payment_verification", [r"\bchecked my bank statement\b", r"\bchecking my bank statement\b", r"\bchecked my account\b", r"\bverified the transaction\b"]),
    ("payment_verified", [r"\bonly one successful transaction\b", r"\bonly one (?:successful )?(?:charge|payment|transaction)\b", r"\bno duplicate (?:charge|payment|transaction)\b"]),
    ("issue_resolved", [r"\beverything is (?:correct|okay|ok|fine)\b", r"\bissue is resolved\b", r"\bproblem is resolved\b", r"\bthat solves it\b", r"\bsorry for the confusion\b"]),
    ("track_refund", [r"\bwhere is my refund\b", r"\btrack (?:my )?refund\b", r"\brefund (?:hasn't|has not) arrived\b"]),
    ("get_refund", [r"\brequest (?:a )?refund\b", r"\bwant (?:a )?refund\b", r"\bplease refund\b"]),
    ("track_order", [r"\bwhere is my order\b", r"\btrack (?:my )?order\b", r"\border status\b"]),
    ("cancel_order", [r"\bcancel (?:my )?order\b", r"\bwant to cancel\b"]),
    ("payment_issue", [r"\bpayment failed\b", r"\bpayment issue\b", r"\bcharged incorrectly\b", r"\bwrong amount (?:was )?charged\b"]),
]


def load_intent_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Intent model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _rule_intent(text):
    for label, patterns in RULE_INTENTS:
        if any(re.search(pattern, text or "", re.IGNORECASE) for pattern in patterns):
            return label
    return None


def predict_message_intents(messages, model):
    customer_messages = [m for m in messages if m.get("speaker", "").lower() == "customer"]
    if not customer_messages:
        return []
    texts = [m.get("message", "") for m in customer_messages]
    model_predictions = model.predict(texts)
    results = []
    for msg, prediction in zip(customer_messages, model_predictions):
        text = msg.get("message", "")
        rule_label = _rule_intent(text)
        label = rule_label or str(prediction)
        confidence, top = None, []
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba([text])[0]
                classes = getattr(model, "classes_", [])
                ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
                top = [{"intent": str(k), "probability": round(float(v) * 100, 2)} for k, v in ranked]
                confidence = round(float(max(probs)) * 100, 2) if len(probs) else None
            except (ValueError, AttributeError, TypeError):
                pass
        if rule_label:
            confidence = 95.0
            top = [{"intent": rule_label, "probability": 95.0}] + [x for x in top if x["intent"] != rule_label][:2]
        results.append({"message_id": msg.get("message_id"), "sequence_number": msg.get("sequence_number"),
                        "message": text, "predicted_intent": label, "confidence": confidence,
                        "top_3_intents": top, "classification_source": "hybrid_rule" if rule_label else "machine_learning"})
    return results


def get_intent_summary(intent_results):
    if not intent_results:
        return {"main_intent": None, "latest_intent": None, "primary_intent": None,
                "total_classified_messages": 0, "intent_timeline": []}
    # Ignore conversational follow-up labels when determining the underlying customer issue.
    non_issue = {"payment_verification", "payment_verified", "issue_resolved"}
    primary = next((x["predicted_intent"] for x in intent_results if x.get("predicted_intent") not in non_issue),
                   intent_results[0].get("predicted_intent"))
    latest = intent_results[-1].get("predicted_intent")
    return {"main_intent": primary, "primary_intent": primary, "latest_intent": latest,
            "total_classified_messages": len(intent_results),
            "intent_timeline": [{"message_id": x.get("message_id"), "intent": x.get("predicted_intent"),
                                 "source": x.get("classification_source", "machine_learning")} for x in intent_results]}
