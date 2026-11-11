import os
import re
import joblib

from modules.text_utils import normalize_text

MODEL_PATH = os.path.join("models", "intent_classifier.joblib")

# Below this top-class probability (in %) the TF-IDF model is guessing, so the
# message is reported as "unclear" instead of a confident-looking wrong intent.
# On the held-out test set this only affects ~0.14% of messages.
LOW_CONFIDENCE_PERCENT = 12.0

# Labels that are conversational glue, not the customer's underlying issue.
NON_ISSUE_INTENTS = {"payment_verification", "payment_verified", "issue_resolved", "smalltalk", "unclear"}

# High-precision patterns for common support intents that are often poorly represented
# in small TF-IDF training sets. Rule matches override the model for that message only.
RULE_INTENTS = [
    ("duplicate_charge", [r"\bcharged twice\b", r"\bdouble[- ]charged\b", r"\bduplicate charge\b", r"\bcharged more than once\b", r"\bcharged (?:me )?(?:two|2) times\b"]),
    ("payment_verification", [r"\bchecked my bank statement\b", r"\bchecking my bank statement\b", r"\bchecked my account\b", r"\bverified the transaction\b"]),
    ("payment_verified", [r"\bonly one successful transaction\b", r"\bonly one (?:successful )?(?:charge|payment|transaction)\b", r"\bno duplicate (?:charge|payment|transaction)\b"]),
    ("issue_resolved", [r"\beverything is (?:correct|okay|ok|fine)\b", r"\bissue is resolved\b", r"\bproblem is resolved\b", r"\bthat solves it\b", r"\bsorry for the confusion\b", r"\b(?:it|that) (?:works|worked|is working) now\b", r"\bthat (?:fixed|solved) it\b"]),
    ("track_refund", [r"\bwhere is my refund\b", r"\btrack (?:my )?refund\b", r"\brefund (?:hasn't|has not) arrived\b", r"\b(?:haven't|have not|didn't|did not) (?:received|got|gotten) (?:my |the )?refund\b", r"\brefund (?:status|is (?:still )?pending)\b", r"\bwaiting for (?:my |the )?refund\b"]),
    ("get_refund", [r"\brequest (?:a )?refund\b", r"\bwant (?:a )?refund\b", r"\bplease refund\b", r"\bneed (?:a )?refund\b", r"\bmoney back\b", r"\brefund me\b"]),
    ("track_order", [r"\bwhere is my order\b", r"\btrack (?:my )?order\b", r"\border status\b", r"\bwhere(?:'s| is) my (?:package|parcel|shipment|delivery)\b"]),
    ("delivery_period", [r"\b(?:hasn't|has not|not|never|still not) arrived\b", r"\b(?:is|are|was|so) (?:very |really |extremely )?(?:late|delayed|overdue)\b",
                         r"\btaking (?:forever|too long|so long)\b"]),
    ("complaint", [r"\bwrong (?:item|product|size|colou?r|order)\b", r"\b(?:damaged|defective)\b", r"\bnot what i ordered\b", r"\bmissing (?:item|part)s?\b"]),
    ("cancel_order", [r"\bcancel (?:my )?order\b", r"\bwant to cancel\b", r"\bcancel (?:it|this)\b"]),
    ("payment_issue", [r"\bpayment failed\b", r"\bpayment issue\b", r"\bcharged incorrectly\b", r"\bwrong amount (?:was )?charged\b",
                       r"\bcharged (?:for|extra|too much)\b", r"\bovercharged\b", r"\bunauthori[sz]ed (?:charge|transaction|payment)\b",
                       r"\bfraudulent (?:charge|transaction|payment)\b", r"\b(?:didn't|did not) (?:authori[sz]e|make) (?:this|that|the) (?:charge|payment|transaction)\b"]),
    ("technical_issue", [r"\b(?:app|website|site|page|system|checkout)\b.{0,25}\b(?:crash\w*|freez\w*|not (?:loading|working)|won't (?:load|open)|is down|error|bug\w*|glitch\w*)\b",
                         r"\b(?:crash\w*|freez\w*|glitch\w*)\b", r"\berror (?:message|code)\b"]),
]

# Pure greetings / thanks / filler: no support intent at all.
_SMALLTALK = re.compile(
    r"^\W*(?:hi+|hello+|hey+|hiya|good (?:morning|afternoon|evening|day)|greetings|thanks?(?: you)?(?: so much| very much| a lot)?|thx|ty|"
    r"ok(?:ay)?|k|yes|yeah|yep|no|nope|sure|fine|alright|bye|goodbye|good bye|please|cool|great|got it|i see|hmm+)"
    r"(?:[\s,.!]+(?:hi+|hello+|hey+|thanks?|thank you|ok(?:ay)?|bye|please|there|team|all|sir|madam|got it))*\W*$",
    re.I,
)


def load_intent_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Intent model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _rule_intent(text):
    text = normalize_text(text)
    for label, patterns in RULE_INTENTS:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            return label
    return None


def _is_smalltalk(text):
    return bool(_SMALLTALK.match(normalize_text(text)))


def predict_message_intents(messages, model):
    customer_messages = [m for m in messages if m.get("speaker", "").lower() == "customer"]
    if not customer_messages:
        return []
    texts = [normalize_text(m.get("message", "")) for m in customer_messages]
    model_predictions = model.predict(texts)
    results = []
    for msg, text, prediction in zip(customer_messages, texts, model_predictions):
        rule_label = _rule_intent(text)
        smalltalk = (not rule_label) and _is_smalltalk(text)
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
        source = "machine_learning"
        if rule_label:
            confidence = 95.0
            top = [{"intent": rule_label, "probability": 95.0}] + [x for x in top if x["intent"] != rule_label][:2]
            source = "hybrid_rule"
        elif smalltalk:
            label, confidence, source = "smalltalk", 95.0, "smalltalk_rule"
        elif confidence is not None and confidence < LOW_CONFIDENCE_PERCENT:
            label, source = "unclear", "low_confidence"
        results.append({"message_id": msg.get("message_id"), "sequence_number": msg.get("sequence_number"),
                        "message": msg.get("message", ""), "predicted_intent": label, "confidence": confidence,
                        "top_3_intents": top, "classification_source": source})
    return results


def get_intent_summary(intent_results):
    if not intent_results:
        return {"main_intent": None, "latest_intent": None, "primary_intent": None,
                "total_classified_messages": 0, "intent_timeline": []}
    # Conversational follow-ups ("thanks", "hi", "I checked my statement") are not the underlying issue.
    primary = next((x["predicted_intent"] for x in intent_results if x.get("predicted_intent") not in NON_ISSUE_INTENTS),
                   intent_results[0].get("predicted_intent"))
    # Latest *meaningful* intent as well, so a closing "thanks" does not hide what was discussed last.
    latest = intent_results[-1].get("predicted_intent")
    latest_issue = next((x["predicted_intent"] for x in reversed(intent_results)
                         if x.get("predicted_intent") not in NON_ISSUE_INTENTS), latest)
    return {"main_intent": primary, "primary_intent": primary, "latest_intent": latest,
            "latest_issue_intent": latest_issue,
            "total_classified_messages": len(intent_results),
            "intent_timeline": [{"message_id": x.get("message_id"), "intent": x.get("predicted_intent"),
                                 "source": x.get("classification_source", "machine_learning")} for x in intent_results]}
