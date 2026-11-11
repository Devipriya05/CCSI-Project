from __future__ import annotations

import inspect
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project paths and optional decision-intelligence modules
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "intent_classifier.joblib"
MODULES_DIR = BASE_DIR / "modules"
if str(MODULES_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(MODULES_DIR))
if str(BASE_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(BASE_DIR))

st.set_page_config(
    page_title="Conversational AI Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# The nine modules are imported independently so one optional import problem
# is reported without preventing the original dashboard from starting.
MODULE_NAMES = (
    "context_manager", "conversation_history", "intent_recognition",
    "emotion_analysis", "emotion_progression", "claim_consistency",
    "escalation_assessment", "agent_routing", "case_summarization",
)
MODULES: dict[str, Any] = {}
MODULE_ERRORS: dict[str, str] = {}
for _name in MODULE_NAMES:
    try:
        MODULES[_name] = __import__(_name)
    except Exception as _exc:
        MODULE_ERRORS[_name] = f"{type(_exc).__name__}: {_exc}"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {"value": value}


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, dict):
        # Common result wrappers
        for key in ("messages", "results", "claims", "emotions", "timeline", "data", "items"):
            if isinstance(value.get(key), (list, tuple)):
                return list(value[key])
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def safe_call(module_name: str, function_name: str, *args, **kwargs):
    """Call a module function while filtering kwargs to its declared signature."""
    module = MODULES.get(module_name)
    fn = getattr(module, function_name, None) if module else None
    if not callable(fn):
        return None
    try:
        sig = inspect.signature(fn)
        parameters = list(sig.parameters.values())
        accepts_varargs = any(p.kind == p.VAR_POSITIONAL for p in parameters)
        positional_capacity = sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in parameters
        )
        call_args = args if accepts_varargs else args[:positional_capacity]
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in parameters)
        positional_names = {
            p.name for p in parameters[:len(call_args)]
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        }
        filtered = kwargs if accepts_kwargs else {
            k: v for k, v in kwargs.items()
            if k in sig.parameters and k not in positional_names
        }
        return fn(*call_args, **filtered)
    except Exception as exc:
        st.warning(f"{module_name}.{function_name} could not complete: {type(exc).__name__}: {exc}")
        return None


def normalize_messages(raw_messages: Any, conversation_id: str) -> list[dict]:
    normalized = []
    for i, item in enumerate(as_list(raw_messages), start=1):
        d = as_dict(item)
        speaker = str(d.get("speaker", d.get("role", d.get("sender", "customer")))).strip()
        message = str(d.get("message", d.get("text", d.get("content", ""))) or "").strip()
        if not message:
            continue
        normalized.append({
            **d,
            "conversation_id": d.get("conversation_id", conversation_id),
            "message_id": d.get("message_id", f"{conversation_id}_MSG{i:03d}"),
            "sequence_number": d.get("sequence_number", d.get("sequence", i)),
            "speaker": speaker,
            "message": message,
            "timestamp": d.get("timestamp", ""),
        })
    return normalized


def parse_input(text: str, conversation_id: str) -> list[dict]:
    parsed = safe_call(
        "context_manager", "parse_conversation", text,
        conversation_id=conversation_id,
    )
    if parsed is not None:
        msgs = normalize_messages(parsed, conversation_id)
        if msgs:
            return msgs

    # Fallback accepts "Customer: ...", "Agent: ..." and unlabelled lines.
    result = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\s*(customer|agent|assistant|support|user)\s*:\s*(.+)$", line, re.I)
        if match:
            speaker, message = match.group(1).lower(), match.group(2).strip()
        else:
            speaker, message = "customer", line
        result.append({"conversation_id": conversation_id,
                       "message_id": f"{conversation_id}_MSG{len(result)+1:03d}",
                       "sequence_number": len(result)+1,
                       "speaker": speaker, "message": message, "timestamp": ""})
    return result


def is_customer(msg: dict) -> bool:
    speaker = str(msg.get("speaker", "customer")).lower()
    return speaker not in {"agent", "assistant", "support", "bot", "system"}


def extract_entities(text: str) -> dict:
    order_ids = re.findall(r"\b(?:order\s*#?\s*|#)([A-Z0-9-]{4,})\b", text or "", re.I)
    emails = re.findall(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text or "", re.I)
    return {"order_ids": list(dict.fromkeys(order_ids)),
            "emails": list(dict.fromkeys(emails))}


NEGATIVE_WORDS = {
    "angry", "disappointed", "frustrated", "terrible", "awful",
    "unacceptable", "upset", "annoyed", "worst", "hate", "ridiculous", "useless"
}
POSITIVE_WORDS = {
    "thank", "thanks", "great", "excellent", "happy",
    "helpful", "wonderful", "appreciate", "resolved"
}
URGENCY_WORDS = {"urgent", "immediately", "asap", "emergency", "right now", "seriously"}
ESCALATION_WORDS = {
    "manager", "supervisor", "escalate", "complaint", "legal action",
    "report this", "still unresolved", "third time", "again and again"
}


_NEGATIVE_EMOTIONS = {"anger": 3, "frustration": 2, "sadness": 1, "anxiety": 1}


def rule_sentiment(text: str) -> dict:
    """Sentiment/risk indicators derived from the emotion_analysis module.

    Falls back to the original small keyword list if that module is missing.
    """
    module = MODULES.get("emotion_analysis")
    analyze = getattr(module, "analyze_emotion", None) if module else None
    if not callable(analyze):
        return _legacy_rule_sentiment(text)
    try:
        result = analyze(text)
    except Exception:
        return _legacy_rule_sentiment(text)

    emotion = result.get("emotion", "neutral")
    evidence = list(result.get("evidence", []))
    legacy = _legacy_rule_sentiment(text)          # keeps escalation/urgency vocabulary
    urgency = list(dict.fromkeys(legacy["urgency_hits"] + (["urgency wording"] if result.get("urgent") else [])))
    escalation = legacy["escalation_hits"]
    negative_points = _NEGATIVE_EMOTIONS.get(emotion, 0)
    positive = emotion == "satisfaction"
    label = "Negative" if negative_points else "Positive" if positive else "Neutral / Uncertain"
    risk_points = negative_points + min(len(urgency), 1) + min(len(escalation), 2)
    return {
        "label": label,
        "positive_words": evidence if positive else [],
        "negative_words": evidence if negative_points else [],
        "urgency_hits": urgency, "escalation_hits": escalation,
        "risk_points": risk_points,
        "risk_level": "High" if risk_points >= 3 else "Medium" if risk_points >= 1 else "Low",
        "score": 1 if positive else -negative_points,
        "emotion": emotion,
    }


def _legacy_rule_sentiment(text: str) -> dict:
    """Original transparent keyword sentiment/risk indicators."""
    normalized = (text or "").lower().replace("’", "'")
    words = re.findall(r"\b[\w']+\b", normalized)
    negative = [word for word in words if word in NEGATIVE_WORDS]
    positive = [word for word in words if word in POSITIVE_WORDS]
    urgency = [term for term in URGENCY_WORDS if term in normalized]
    escalation = [term for term in ESCALATION_WORDS if term in normalized]

    if len(negative) > len(positive):
        label = "Negative"
    elif len(positive) > len(negative):
        label = "Positive"
    else:
        label = "Neutral / Uncertain"

    risk_points = min(len(negative), 2) + min(len(urgency), 1) + min(len(escalation), 2)
    risk_level = "High" if risk_points >= 3 else "Medium" if risk_points >= 1 else "Low"
    return {
        "label": label, "positive_words": positive, "negative_words": negative,
        "urgency_hits": urgency, "escalation_hits": escalation,
        "risk_points": risk_points, "risk_level": risk_level,
        "score": len(positive) - len(negative),
    }



def detect_resolution(messages):
    """Single source of truth: context_manager.detect_resolution_status.

    Earlier the dashboard and the module pipeline used two different
    implementations that could disagree on the same chat.  The legacy
    implementation below is only a fallback if the module cannot be imported.
    """
    module = MODULES.get("context_manager")
    fn = getattr(module, "detect_resolution_status", None) if module else None
    if callable(fn):
        try:
            result = fn(messages)
            evidence = result.get("evidence") or []
            if isinstance(evidence, str):
                evidence = [f'Latest customer message: "{evidence}"'] if evidence else []
            return {
                "status": result.get("status", "Not confirmed"),
                "resolved": bool(result.get("resolved", False)),
                "evidence": evidence,
                "next_action": result.get("next_action", "Follow up with the customer to confirm whether the issue is resolved."),
            }
        except Exception:
            pass
    return _legacy_detect_resolution(messages)


def _legacy_detect_resolution(messages):
    """Conservative, evidence-based resolution detection (fallback)."""

    customer_texts = [
        m["message"].strip()
        for m in messages
        if is_customer(m)
    ]

    if not customer_texts:
        return {
            "status": "Unknown",
            "resolved": False,
            "evidence": [],
            "next_action": "Review the conversation."
        }

    latest = customer_texts[-1]
    latest_lower = latest.lower()

    # Detect statements that indicate the issue is still outstanding.
    unresolved_patterns = [
        r"\bstill\s+(?:waiting|not received|haven't|have not|not arrived|not delivered)\b",
        r"\bnot\s+(?:arrived|received|delivered|resolved|fixed)\b",
        r"\bhasn't\s+arrived\b",
        r"\bhave not received\b",
        r"\bremains?\s+unresolved\b",
        r"\bstill\s+not\s+resolved\b",
        r"\bno\s+one\s+has\s+helped\b",
        r"\bnot yet (?:received|resolved|fixed|credited)\b",
        r"\bwaiting for (?:my|the) refund\b",
    ]

    # Explicit resolution statements.
    resolved_patterns = [
        r"\b(?:i\s+)?received\s+(?:the\s+)?(?:package|parcel|order|refund)\b",
        r"\b(?:package|parcel|order)\s+(?:has\s+)?arrived\b",
        r"\b(?:it|the issue|the problem)\s+(?:is\s+)?(?:resolved|fixed)\b",
        r"\b(?:this|that)\s+is\s+(?:resolved|sorted)\b",
        r"\brefund\s+(?:has\s+)?(?:arrived|been received|been credited)\b",
        r"\bactually[, ]+(?:i\s+)?(?:received|got|have it)\b",
        r"\bproblem\s+(?:is\s+)?solved\b",
        r"\beverything\s+is\s+(?:fine|correct|okay|ok)\s+now\b",
        r"\bno\s+further\s+(?:help|assistance)\s+(?:is\s+)?needed\b",
    ]

    # Customer withdrawal / correction of an earlier complaint.
    withdrawal_patterns = [
        r"\bsorry for the confusion\b",
        r"\bmy mistake\b",
        r"\bi was mistaken\b",
        r"\bi was wrong\b",
        r"\bthere (?:is|was) only one successful transaction\b",
        r"\bonly one transaction (?:was )?successful\b",
        r"\bi checked (?:my )?(?:bank statement|account|payment details)\b",
        r"\bthere was no duplicate charge\b",
        r"\bi was not charged twice\b",
        r"\bno duplicate (?:payment|charge|transaction)\b",
    ]

    unresolved_matches = [
        p for p in unresolved_patterns
        if re.search(p, latest_lower, re.I)
    ]

    resolved_matches = [
        p for p in resolved_patterns
        if re.search(p, latest_lower, re.I)
    ]

    withdrawal_matches = [
        p for p in withdrawal_patterns
        if any(re.search(p, text.lower(), re.I) for text in customer_texts)
    ]

    # Outstanding issue takes precedence.
    if unresolved_matches:
        return {
            "status": "Unresolved",
            "resolved": False,
            "evidence": [
                f'Latest customer message indicates an outstanding issue: "{latest}"'
            ],
            "next_action": (
                "Continue investigating the outstanding issue "
                "and provide an update."
            )
        }

    # Explicit resolution.
    if resolved_matches:
        return {
            "status": "Explicitly resolved",
            "resolved": True,
            "evidence": [
                f'Latest customer message confirms resolution: "{latest}"'
            ],
            "next_action": (
                "Confirm whether further assistance is needed; "
                "if no additional issue remains, close the case."
            )
        }

    # Withdrawal requires both a correction/verification and an apology
    # or explicit withdrawal in the conversation.
    verification_patterns = [
        r"\bthere (?:is|was) only one successful transaction\b",
        r"\bonly one transaction (?:was )?successful\b",
        r"\bi checked (?:my )?(?:bank statement|account|payment details)\b",
        r"\bthere was no duplicate charge\b",
        r"\bi was not charged twice\b",
        r"\bno duplicate (?:payment|charge|transaction)\b",
    ]

    withdrawal_confirmation_patterns = [
        r"\bsorry for the confusion\b",
        r"\bmy mistake\b",
        r"\bi was mistaken\b",
        r"\bi was wrong\b",
        r"\beverything is correct\b",
    ]

    verified = any(
        re.search(pattern, text.lower(), re.I)
        for pattern in verification_patterns
        for text in customer_texts
    )

    acknowledged = any(
        re.search(pattern, text.lower(), re.I)
        for pattern in withdrawal_confirmation_patterns
        for text in customer_texts
    )

    if verified and acknowledged:
        return {
            "status": "Resolved - complaint withdrawn",
            "resolved": True,
            "evidence": [
                "Customer verified that only one successful transaction existed.",
                "Customer acknowledged the earlier complaint was a confusion or mistake."
            ],
            "next_action": (
                "Mark the duplicate-charge complaint as withdrawn "
                "and close the case if no other issue remains."
            )
        }

    return {
        "status": "Not confirmed",
        "resolved": False,
        "evidence": [],
        "next_action": (
            "Follow up with the customer to confirm whether "
            "the issue is resolved."
        )
    }

def priority_from_score(score: float) -> str:
    """Same 0-100 scale as modules/escalation_assessment.py."""
    if score >= 60:
        return "Critical"
    if score >= 35:
        return "High"
    if score >= 20:
        return "Medium"
    return "Low"


def flatten_text(messages: list[dict]) -> str:
    return "\n".join(f'{m["speaker"].title()}: {m["message"]}' for m in messages)


# ---------------------------------------------------------------------------
# Original conversation-level model + hybrid correction layer
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading intent classifier…")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def model_predict(model, text: str) -> tuple[str, float, list[tuple[str, float]]]:
    if not text.strip():
        return "unknown", 0.0, []
    predicted = str(model.predict([text])[0])
    candidates = []
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba([text])[0]
            classes = list(getattr(model, "classes_", []))
            candidates = sorted(
                [(str(classes[i]), float(probs[i])) for i in range(min(len(classes), len(probs)))],
                key=lambda x: x[1], reverse=True
            )
            confidence = next((p for label, p in candidates if label == predicted), max((p for _, p in candidates), default=0.0))
        except Exception:
            pass
    return predicted, confidence, candidates


def hybrid_correct(message: str, model_intent: str) -> tuple[str, str]:
    """Preserve the original app's hybrid rules, correcting malformed regexes."""
    text = (message or "").lower().strip()

    refund_terms = ["refund", "reimbursement", "compensation", "restitution", "rebate"]
    refund_phrases = [
        "track my refund", "track the refund", "refund status",
        "status of my refund", "where is my refund", "where's my refund",
        "track my reimbursement", "status of my reimbursement",
    ]
    if any(term in text for term in refund_terms) and any(p in text for p in refund_phrases):
        return "track_refund", "Rule-based correction"

    # Explicit newsletter/order contrasts. The original source contained
    # cancel\\w\\*; the intended regex is cancel\\w*.
    cancel_newsletter = re.search(r"\bcancel\w*\b.{0,35}\b(newsletter|subscription)\b", text)
    not_order = re.search(r"\bnot\b.{0,20}\b(order|purchase|shipment)\b", text)
    if cancel_newsletter and not_order:
        return "newsletter_subscription", "Rule-based correction"

    cancel_order = re.search(r"\bcancel\w*\b.{0,35}\b(order|purchase|shipment)\b", text)
    not_newsletter = re.search(r"\bnot\b.{0,20}\bnewsletter\b", text)
    if cancel_order and not_newsletter:
        return "cancel_order", "Rule-based correction"

    negation_patterns = [
        r"\bdo not cancel\b", r"\bdon't cancel\b", r"\bdont cancel\b",
        r"\bdo not want to cancel\b", r"\bdon't want to cancel\b",
        r"\bdo not wish to cancel\b", r"\bdon't wish to cancel\b",
    ]
    if any(re.search(pattern, text) for pattern in negation_patterns):
        return model_intent, "Machine learning"

    order_cancel_patterns = [
        r"\bcancel\w*\b.{0,35}\b(order|purchase|shipment)\b",
        r"\b(order|purchase|shipment)\b.{0,35}\bcancel\w*\b",
        r"\bcall off\b.{0,25}\b(order|purchase|shipment)\b",
    ]
    if any(re.search(pattern, text) for pattern in order_cancel_patterns):
        return "cancel_order", "Rule-based correction"

    delivery_terms = [
        "delivery date", "delivery time", "delivery period", "delivery delay",
        "package is late", "package arrived late", "delivery is late",
        "shipment is delayed", "package is delayed", "when will my package arrive",
        "when will my order arrive", "when will my parcel arrive",
        "when should my package arrive", "expected delivery date",
    ]
    has_refund_context = any(term in text for term in refund_terms) or any(
        term in text for term in ["money back", "money returned", "get my money"]
    )
    if any(term in text for term in delivery_terms) and not has_refund_context:
        return "delivery_period", "Rule-based correction"

    tracking_phrases = [
        "track my order", "track the order", "track my package", "track the package",
        "track my parcel", "track the parcel", "tracking my order", "tracking my package",
        "where is my order", "where's my order", "where is my package",
        "where's my package", "where is my parcel", "where's my parcel",
        "order status", "package status", "parcel status",
    ]
    if any(phrase in text for phrase in tracking_phrases) and not has_refund_context:
        return "track_order", "Rule-based correction"

    return model_intent, "Machine learning"


def sentiment_details(messages: list[dict]) -> dict:
    rows = []
    all_positive, all_negative = [], []
    for m in messages:
        if is_customer(m):
            s = rule_sentiment(m["message"])
            rows.append({"Message": m["message"], "Sentiment": s["label"],
                         "Positive indicators": ", ".join(s["positive_words"]),
                         "Negative indicators": ", ".join(s["negative_words"])})
            all_positive += s["positive_words"]
            all_negative += s["negative_words"]
    return {"rows": rows, "positive": all_positive, "negative": all_negative}


def risk_assessment(messages: list[dict], escalation: dict | None = None, resolution: dict | None = None) -> dict:
    customer_text = " ".join(m["message"].lower() for m in messages if is_customer(m))
    customer_text = customer_text.replace("’", "'")
    urgency_terms = ["urgent", "immediately", "asap", "unacceptable", "lawyer", "legal", "complaint",
                     "manager", "supervisor", "escalate", "refund now", "chargeback", "never again",
                     "scam", "fraud", "sue ", "ombudsman", "worst"]
    hits = [term.strip() for term in urgency_terms if term in customer_text]
    negative_points = sum(
        _NEGATIVE_EMOTIONS.get(rule_sentiment(m["message"]).get("emotion"), 0)
        for m in messages if is_customer(m)
    )
    # Fallback estimate, only used if the escalation module produced nothing.
    score = min(100, len(hits) * 12 + negative_points * 7)
    if resolution and resolution.get("resolved"):
        score = min(score, 19)
    if escalation:
        try:
            # Single source of truth: modules/escalation_assessment.py (0-100).
            score = float(escalation.get("risk_score", escalation.get("escalation_score", escalation.get("score", score))))
        except (TypeError, ValueError):
            pass
    return {"score": int(min(100, score)), "level": priority_from_score(score), "keywords": hits}


def call_optional_pipeline(messages: list[dict], conversation_id: str, customer_id: str):
    """Run all nine modules using the documented interfaces."""
    context = safe_call("context_manager", "get_conversation_summary", messages)
    intent_model = safe_call("intent_recognition", "load_intent_model")
    intent_results = safe_call("intent_recognition", "predict_message_intents", messages, intent_model) if intent_model is not None else None
    intent_summary = safe_call("intent_recognition", "get_intent_summary", intent_results) if intent_results is not None else None

    emotion_results = safe_call("emotion_analysis", "analyze_customer_emotions", messages)
    emotion_summary = safe_call("emotion_analysis", "get_emotion_summary", emotion_results) if emotion_results is not None else None
    progression = safe_call("emotion_progression", "analyze_emotion_progression", emotion_results) if emotion_results is not None else None
    claims = safe_call("claim_consistency", "analyze_claim_consistency", messages)

    history_mod = MODULES.get("conversation_history")
    contact_stats = None
    if history_mod:
        safe_call("conversation_history", "initialize_database")
        contact_stats = safe_call("conversation_history", "get_contact_statistics", customer_id)
    escalation = safe_call("escalation_assessment", "assess_escalation",
                           messages, emotion_results, progression, contact_stats, intent_summary)
    routing = safe_call("agent_routing", "recommend_agent", intent_summary, escalation)
    case_summary = safe_call("case_summarization", "generate_case_summary",
        customer_id, messages, intent_summary, emotion_results, progression,
        contact_stats, claims, escalation, routing)
    return {
        "context": context, "intent_results": intent_results, "intent_summary": intent_summary,
        "emotion_results": emotion_results, "emotion_summary": emotion_summary,
        "progression": progression, "claims": claims, "contact_stats": contact_stats,
        "escalation": escalation, "routing": routing, "case_summary": case_summary,
    }


# ---------------------------------------------------------------------------
# Safe SQLite history adapter. Existing module functions are preferred.
# Direct fallback is deliberately non-destructive and creates a separate
# integration table only when no compatible save API is exposed.
# ---------------------------------------------------------------------------
def save_history(conversation_id: str, customer_id: str, messages: list[dict]) -> tuple[bool, str]:
    mod = MODULES.get("conversation_history")
    if not mod:
        return False, "conversation_history module is unavailable."
    init = getattr(mod, "initialize_database", None)
    if callable(init):
        try:
            init()
        except Exception as exc:
            return False, f"Database initialization failed: {exc}"

    save_fn = getattr(mod, "save_conversation", None)
    if callable(save_fn):
        payload = {"conversation_id": conversation_id, "customer_id": customer_id,
                   "messages": messages, "conversation_text": flatten_text(messages)}
        try:
            sig = inspect.signature(save_fn)
            params = list(sig.parameters.values())
            if any(p.kind == p.VAR_KEYWORD for p in params):
                save_fn(**payload)
            else:
                kwargs = {k: v for k, v in payload.items() if k in sig.parameters}
                required = [p for p in params if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                if len(kwargs) == len(required) or all(p.name in kwargs for p in required):
                    save_fn(**kwargs)
                elif len(params) >= 3:
                    save_fn(conversation_id, customer_id, messages)
                elif len(params) == 2:
                    save_fn(customer_id, messages)
                else:
                    save_fn(messages)
            return True, "Conversation saved using the existing history module."
        except Exception as exc:
            return False, f"Existing save_conversation() failed: {type(exc).__name__}: {exc}"

    db_path = getattr(mod, "DB_PATH", BASE_DIR / "conversation_history.db")
    db_path = Path(db_path)
    if not db_path.is_absolute():
        db_path = BASE_DIR / db_path
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS app_saved_conversations (
                conversation_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL,
                conversation_text TEXT NOT NULL, saved_at TEXT NOT NULL)""")
            conn.execute("""INSERT OR IGNORE INTO app_saved_conversations
                (conversation_id, customer_id, conversation_text, saved_at)
                VALUES (?, ?, ?, ?)""",
                (conversation_id, customer_id, flatten_text(messages), datetime.now().isoformat(timespec="seconds")))
        return True, "Conversation saved in the non-destructive integration history table."
    except Exception as exc:
        return False, f"Could not save conversation: {type(exc).__name__}: {exc}"


def load_history(customer_id: str) -> pd.DataFrame:
    mod = MODULES.get("conversation_history")
    if not mod:
        return pd.DataFrame()
    fn = getattr(mod, "get_customer_history", None)
    if callable(fn):
        try:
            result = fn(customer_id)
            if isinstance(result, pd.DataFrame):
                return result
            return pd.DataFrame(as_list(result))
        except Exception:
            pass
    db_path = getattr(mod, "DB_PATH", BASE_DIR / "conversation_history.db")
    db_path = Path(db_path)
    if not db_path.is_absolute():
        db_path = BASE_DIR / db_path
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
            if "app_saved_conversations" in tables:
                return pd.read_sql_query(
                    "SELECT * FROM app_saved_conversations WHERE customer_id=? ORDER BY saved_at DESC",
                    conn, params=(customer_id,))
    except Exception:
        pass
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Modern Dashboard — Customer Support Case Intelligence
# ---------------------------------------------------------------------------

import json

# ------------------------- Styling -------------------------

st.markdown("""
<style>
        /* Sidebar readability */
    [data-testid="stSidebar"] {
        background-color: #F7F9FC !important;
        color: #25344D !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #25344D !important;
    }

    /* Main headings */
    h1, h2, h3, h4,
    .section-heading,
    .brand-name {
        color: #1F2A44 !important;
    }

    /* Better input contrast */
    textarea, input {
        background-color: #FFFFFF !important;
        color: #25344D !important;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------- Small helper functions -------------------------

def get_value(data, keys, default="Not available"):
    """Get the first available value from a dictionary."""
    if not isinstance(data, dict):
        return default

    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value

    return default


def clean_label(value):
    """Make technical labels easier to read."""
    if value is None:
        return "Not available"

    return str(value).replace("_", " ").strip().title()


def get_result_rows(raw_result):
    """Convert module output into a list of dictionaries."""
    rows = as_list(raw_result)
    return [as_dict(row) for row in rows]


def make_annotation_map(rows, possible_labels):
    """
    Create message-level labels using message text or sequence number.
    No conversation-level prediction is falsely assigned to every message.
    """
    mapping = {}

    for index, row in enumerate(rows, start=1):
        label = get_value(row, possible_labels, default=None)

        if label is None:
            continue

        message_text = get_value(
            row,
            ["message", "text", "content"],
            default=None
        )

        sequence = get_value(
            row,
            ["sequence_number", "sequence", "message_number"],
            default=None
        )

        if message_text:
            mapping[str(message_text).strip().lower()] = label

        if sequence is not None:
            mapping[str(sequence)] = label

        mapping[f"index_{index}"] = label

    return mapping


# ------------------------- Sidebar navigation -------------------------

with st.sidebar:

    st.markdown("""
    <div class="brand">
        <div class="brand-icon">✦</div>
        <div>
            <div class="brand-name">Case Intelligence</div>
            <div class="brand-caption">Conversational AI Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    selected_page = st.radio(
        "WORKSPACE",
        [
            "Case Dashboard",
            "Advanced Analysis",
            "Reports & Downloads"
        ],
        label_visibility="visible"
    )

    st.divider()

    if st.button("＋ New Analysis", use_container_width=True):
        st.session_state.pop("latest_analysis", None)
        st.session_state["conversation_input"] = ""
        st.rerun()

    st.caption("Customer Support Case Intelligence")


# ------------------------- Page heading -------------------------

st.markdown("""
<div class="brand">
    <div class="brand-icon">🧠✨</div>
    <div>
        <div class="brand-name">Customer Support Intelligence</div>
        <div class="brand-caption">
            Understand conversations. Detect customer needs. Support better decisions.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------- Conversation input -------------------------

if selected_page == "Case Dashboard":

    st.markdown(
        '<div class="section-heading">Analyze a conversation</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Paste a customer-support conversation below. "
        "Use Customer: and Agent: to identify the speakers."
    )

    conversation_text = st.text_area(
        "Conversation",
        height=180,
        key="conversation_input",
        placeholder=(
            "Customer: I think I was charged twice for my order.\n"
            "Agent: I apologize. Could you please confirm the payment details?\n"
            "Customer: I checked my bank statement again.\n"
            "Customer: There is only one successful transaction."
        ),
        label_visibility="collapsed"
    )

    analyze_clicked = st.button(
        "✦ Analyze Conversation",
        type="primary",
        use_container_width=True
    )

    if analyze_clicked:

        if not conversation_text.strip():

            st.warning("Please enter a conversation before starting the analysis.")

        else:

            customer_id = "ANONYMOUS"
            conversation_id = f"CASE-{datetime.now():%Y%m%d-%H%M%S}"

            messages = parse_input(
                conversation_text,
                conversation_id
            )

            if not messages:

                st.warning("No valid messages could be extracted.")

            else:

                with st.spinner("Analyzing conversation intelligence..."):

                    customer_messages = [
                        message for message in messages
                        if is_customer(message)
                    ]

                    full_text = flatten_text(messages)

                    # The intent is what the CUSTOMER asked for. Agent lines
                    # ("let me check your order status") and the speaker
                    # prefixes must not feed the classifier or the rules.
                    customer_text = " ".join(
                        message["message"] for message in customer_messages
                    ) or full_text

                    # Main trained intent model
                    model_error = None

                    try:
                        model = load_model()

                        model_intent, model_confidence, candidates = (
                            model_predict(model, customer_text)
                        )

                    except Exception as exc:

                        model_intent = "unknown"
                        model_confidence = 0.0
                        candidates = []

                        model_error = (
                            f"{type(exc).__name__}: {exc}"
                        )

                    final_intent, correction_reason = hybrid_correct(
                        customer_text,
                        model_intent
                    )

                    low_confidence = model_confidence < 0.40

                    # Existing nine-module intelligence pipeline
                    results = call_optional_pipeline(
                        messages,
                        conversation_id,
                        customer_id
                    )

                    resolution = detect_resolution(messages)

                    escalation = (
                        as_dict(results.get("escalation"))
                        if results.get("escalation") is not None
                        else {}
                    )

                    original_score = escalation.get(
                        "escalation_score",
                        escalation.get("score")
                    )

                    # escalation_assessment already lowers the risk when the
                    # customer reports the issue as resolved, so the dashboard
                    # keeps the module's score, level and handoff unchanged.
                    if False:

                        try:
                            adjusted_score = (
                                min(float(original_score), 25.0)
                                if original_score is not None
                                else 10.0
                            )

                        except (TypeError, ValueError):

                            adjusted_score = 10.0

                        escalation = {
                            **escalation,
                            "original_module_score": original_score,
                            "escalation_score": adjusted_score,
                            "score": adjusted_score,
                            "priority": priority_from_score(adjusted_score),
                            "human_review_required": False,
                            "resolution_adjustment": (
                                "Reduced because the latest customer message "
                                "explicitly confirms resolution."
                            )
                        }

                    else:

                        try:
                            score = (
                                float(original_score)
                                if original_score is not None
                                else 0.0
                            )

                        except (TypeError, ValueError):

                            score = 0.0

                        escalation = {
                            **escalation,
                            "escalation_score": score,
                            "score": score,
                            "priority": escalation.get(
                                "priority",
                                priority_from_score(score)
                            )
                        }

                    risk = risk_assessment(
                        messages,
                        escalation,
                        resolution
                    )

                    sentiment = sentiment_details(messages)

                    entities = extract_entities(full_text)

                    customer_count = len(customer_messages)
                    agent_count = len(messages) - customer_count

                    words = Counter(
                        re.findall(
                            r"\b[a-zA-Z']{3,}\b",
                            full_text.lower()
                        )
                    )

                    frequent_words = pd.DataFrame(
                        words.most_common(12),
                        columns=["Word", "Count"]
                    )

                    # Store all results for the three dashboard pages
                    analysis = {
                        "messages": messages,
                        "model_intent": model_intent,
                        "model_confidence": model_confidence,
                        "candidates": candidates,
                        "final_intent": final_intent,
                        "correction_reason": correction_reason,
                        "low_confidence": low_confidence,
                        "results": results,
                        "resolution": resolution,
                        "escalation": escalation,
                        "risk": risk,
                        "sentiment": sentiment,
                        "entities": entities,
                        "customer_count": customer_count,
                        "agent_count": agent_count,
                        "frequent_words": frequent_words,
                        "model_error": model_error,
                        "full_text": full_text,
                        "customer_id": customer_id,
                        "conversation_id": conversation_id
                    }

                    st.session_state["latest_analysis"] = analysis


# ------------------------- Load latest analysis -------------------------

analysis = st.session_state.get("latest_analysis")


if analysis:

    messages = analysis["messages"]
    results = analysis["results"]

    resolution = analysis["resolution"]
    escalation = analysis["escalation"]
    risk = analysis["risk"]

    customer_id = analysis["customer_id"]
    conversation_id = analysis["conversation_id"]

    routing = (
        as_dict(results.get("routing"))
        if results.get("routing") is not None
        else {}
    )

    claims = (
        as_dict(results.get("claims"))
        if results.get("claims") is not None
        else {}
    )

    emotion_summary = (
        as_dict(results.get("emotion_summary"))
        if results.get("emotion_summary") is not None
        else {}
    )

    progression = (
        as_dict(results.get("progression"))
        if results.get("progression") is not None
        else {}
    )
    # Shared emotion timeline data for all dashboard pages
    timeline = as_list(progression.get("message_timeline"))

    timeline_rows = [
        as_dict(item)
        for item in timeline
    ]

    intent_summary = (
        as_dict(results.get("intent_summary"))
        if results.get("intent_summary") is not None
        else {}
    )

    contact_stats = (
        as_dict(results.get("contact_stats"))
        if results.get("contact_stats") is not None
        else {}
    )

    intent_rows = get_result_rows(results.get("intent_results"))
    # Derive conversation intent from message-level results
    # The module's summary skips greetings / thanks / "I checked my statement"
    # style follow-ups, so a chat starting with "Hi" still gets the real issue.
    primary_message_intent = get_value(
        intent_summary,
        ["primary_intent", "main_intent"],
        default=analysis["final_intent"]
    )

    # If every message was too vague, fall back to the whole-conversation
    # prediction, and admit uncertainty when that is weak as well.
    if str(primary_message_intent).lower() in ("unclear", "smalltalk", "not available"):
        if analysis["model_confidence"] >= 0.12 and analysis["final_intent"] != "unknown":
            primary_message_intent = analysis["final_intent"]
        else:
            primary_message_intent = "unclear"

    latest_message_intent = (
        get_value(
            intent_rows[-1],
            ["predicted_intent", "intent", "label"],
            default="Not available"
        )
        if intent_rows else "Not available"
    )
    emotion_rows = get_result_rows(results.get("emotion_results"))

    intent_map = make_annotation_map(
        intent_rows,
        [
            "predicted_intent",
            "intent",
            "label",
            "predicted_label"
        ]
    )

    emotion_map = make_annotation_map(
        emotion_rows,
        [
            "predicted_emotion",
            "emotion",
            "label"
        ]
    )

    latest_emotion = get_value(
        emotion_summary,
        ["latest_emotion", "dominant_emotion", "overall_emotion"],
        default=get_value(
            progression,
            ["latest_emotion", "peak_emotion"],
            default="Not available"
        )
    )

    if emotion_rows:

        latest_emotion = get_value(
            emotion_rows[-1],
            ["predicted_emotion", "emotion", "label"],
            default=latest_emotion
        )

    escalation_priority = get_value(
        escalation,
        ["priority"],
        default=priority_from_score(
            float(escalation.get("score", 0) or 0)
        )
    )


    # ============================================================
    # PAGE 1 — MAIN CASE DASHBOARD
    # ============================================================

    if selected_page == "Case Dashboard":

        st.divider()

        st.markdown("### Case overview")

        st.caption(
            "A concise view of the conversation's detected customer state "
            "and recommended handling."
        )

        # Four primary cards
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Primary intent",
            clean_label(primary_message_intent)
        )

        c2.metric(
            "Latest emotion",
            clean_label(latest_emotion)
        )

        c3.metric(
            "Resolution",
            clean_label(resolution["status"])
        )

        c4.metric(
            "Escalation priority",
            clean_label(escalation_priority)
        )

        # Secondary overview
        st.markdown("#### Conversation insights")

        a, b, c = st.columns(3)

        a.metric(
            "Total messages",
            len(messages)
        )

        b.metric(
            "Customer messages",
            analysis["customer_count"]
        )

        c.metric(
            "Agent messages",
            analysis["agent_count"]
        )

        st.divider()

        # Conversation timeline
        st.markdown("### Conversation timeline")

        st.caption(
            "Detected intent and emotion labels are shown when "
            "the corresponding message-level modules return results."
        )

        for index, message in enumerate(messages, start=1):

            speaker = str(
                message.get("speaker", "customer")
            ).lower()

            customer_message = is_customer(message)

            chat_role = "user" if customer_message else "assistant"

            message_text = message.get("message", "")

            sequence = str(
                message.get("sequence_number", index)
            )

            # Labels exist for customer messages only. Look up by sequence
            # number first (unique), then by text. The old "index_N"
            # fallback counted customer rows only, so agent bubbles could
            # be tagged with a customer message's emotion/intent.
            if customer_message:
                intent_label = (
                    intent_map.get(sequence)
                    or intent_map.get(message_text.strip().lower())
                )
                emotion_label = (
                    emotion_map.get(sequence)
                    or emotion_map.get(message_text.strip().lower())
                )
            else:
                intent_label = None
                emotion_label = None

            with st.chat_message(chat_role):

                st.caption(
                    "Customer" if customer_message else "Support agent"
                )

                st.write(message_text)

                tags = []

                if intent_label:
                    tags.append(
                        f"Intent: {clean_label(intent_label)}"
                    )

                if emotion_label:
                    tags.append(
                        f"Emotion: {clean_label(emotion_label)}"
                    )

                if tags:
                    st.caption("  ·  ".join(tags))


        st.divider()

        # Customer intelligence and risk
        left, right = st.columns([1.1, 1])

        with left:

            st.markdown("### Customer intelligence")

            st.metric(
                "Risk score",
                f"{risk['score']} / 100"
            )

            st.progress(
                min(max(int(risk["score"]), 0), 100)
            )

            st.write(
                f"**Risk level:** {clean_label(risk['level'])}"
            )

            st.write(
                f"**Handoff signal:** {clean_label(escalation.get('handoff_action', 'Not available'))}"
                f" · **Urgent:** {'Yes' if escalation.get('urgent') else 'No'}"
            )
            if escalation.get("handoff_reason"):
                st.caption(escalation["handoff_reason"])

            st.write(
                f"**Latest emotion:** {clean_label(latest_emotion)}"
            )

            st.write(
                f"**Emotion progression:** "
                f"{clean_label(progression.get('progression', 'Not available'))}"
            )

            risk_keywords = risk.get("keywords", [])

            st.caption(
                "Risk indicators: "
                + (", ".join(risk_keywords) if risk_keywords else "None detected")
            )

        with right:

            st.markdown("### Resolution assessment")

            if resolution["resolved"]:

                st.success(
                    f"Case status: {resolution['status']}"
                )

            else:

                st.warning(
                    f"Case status: {resolution['status']}"
                )

            st.write(resolution["next_action"])

            for evidence in resolution.get("evidence", []):
                st.caption(f"Evidence: {evidence}")

            if analysis["low_confidence"]:

                st.info(
                    "The intent prediction has low confidence. "
                    "A clarification may be needed before taking action."
                )


        st.divider()

        # Emotion progression chart
        st.markdown("### Customer emotion progression")


        timeline_df = pd.DataFrame(timeline_rows)

        if (
            not timeline_df.empty
            and "severity" in timeline_df.columns
        ):

            if "sequence_number" in timeline_df.columns:

                timeline_df = timeline_df.set_index(
                    "sequence_number"
                )

            st.line_chart(
                timeline_df[["severity"]],
                use_container_width=True
            )

            st.caption(
                "Severity values are based on the emotion progression module."
            )

        elif emotion_rows:

            emotion_labels = []

            for row in emotion_rows:

                label = get_value(
                    row,
                    ["predicted_emotion", "emotion", "label"],
                    default="Unknown"
                )

                emotion_labels.append(clean_label(label))

            emotion_counts = (
                pd.Series(emotion_labels)
                .value_counts()
                .rename_axis("Emotion")
                .to_frame("Messages")
            )

            st.bar_chart(
                emotion_counts,
                use_container_width=True
            )

        else:

            st.info(
                "Emotion progression chart is unavailable "
                "because no message-level emotion timeline was returned."
            )


        st.divider()

        # Support recommendation
        st.markdown("### Recommended support action")

        if routing:

            department = get_value(
                routing,
                ["recommended_department", "department"]
            )

            agent_level = get_value(
                routing,
                ["recommended_agent_level", "agent_level"]
            )

            routing_priority = get_value(
                routing,
                ["routing_priority"]
            )

            st.markdown(f"**Recommended department:** {department}")

            st.markdown(f"**Recommended agent level:** {agent_level}")

            st.markdown(f"**Routing priority:** {routing_priority}")

            st.caption(
                "This is an AI-generated recommendation. "
                "No agent has been assigned and no transfer has occurred."
            )

        else:

            st.info(
                "A support routing recommendation is not available for this case."
            )

        case_summary = results.get("case_summary")

        if case_summary is not None:

            with st.expander("View generated case summary"):

                if isinstance(case_summary, dict):

                    st.table(
                        pd.DataFrame(
                            list(case_summary.items()),
                            columns=["Field", "Value"]
                        )
                    )

                else:

                    st.write(str(case_summary))


    # ============================================================
    # PAGE 2 — ADVANCED ANALYSIS
    # ============================================================

    elif selected_page == "Advanced Analysis":

        st.markdown("### Advanced intelligence analysis")

        st.caption(
            "Technical outputs from the intent, emotion, consistency, "
            "escalation and routing modules."
        )

        # Intent model
        with st.expander("Intent recognition model", expanded=True):

            a, b = st.columns(2)

            a.metric(
                "Original model prediction",
                clean_label(analysis["model_intent"])
            )

            b.metric(
                "Model confidence",
                f"{analysis['model_confidence']:.1%}"
            )

            st.write(
                f"**Final hybrid intent:** "
                f"{clean_label(analysis['final_intent'])}"
            )

            st.caption(
                f"Decision source: {analysis['correction_reason']}"
            )

            if analysis["candidates"]:

                candidate_df = pd.DataFrame(
                    analysis["candidates"][:5],
                    columns=["Intent", "Probability"]
                )

                candidate_df = candidate_df.set_index("Intent")

                st.markdown("**Top intent candidates**")

                st.bar_chart(candidate_df)

            if intent_rows:

                st.markdown("**Message-level intent results**")

                st.dataframe(
                    pd.DataFrame(intent_rows),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No message-level intent results were returned."
                )


        # Emotion analysis
        with st.expander("Emotion analysis"):

            if emotion_rows:

                st.dataframe(
                    pd.DataFrame(emotion_rows),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("No message-level emotion results were returned.")

            if emotion_summary:

                st.markdown("**Overall emotion summary**")

                st.table(
                    pd.DataFrame(
                        list(emotion_summary.items()),
                        columns=["Measure", "Result"]
                    )
                )


        # Emotion progression
        with st.expander("Emotion progression details"):

            if progression:

                st.write(
                    f"**Progression:** "
                    f"{clean_label(progression.get('progression', 'Not available'))}"
                )

                st.write(
                    f"**First emotion:** "
                    f"{clean_label(progression.get('first_emotion', 'Not available'))}"
                )

                st.write(
                    f"**Latest emotion:** "
                    f"{clean_label(progression.get('latest_emotion', 'Not available'))}"
                )

                st.write(
                    f"**Peak emotion:** "
                    f"{clean_label(progression.get('peak_emotion', 'Not available'))}"
                )

                if timeline_rows:

                    st.dataframe(
                        pd.DataFrame(timeline_rows),
                        use_container_width=True,
                        hide_index=True
                    )

            else:

                st.info("Emotion progression details are unavailable.")


        # Claim consistency
        with st.expander("Claim consistency analysis"):

            st.metric(
                "Claims detected",
                claims.get(
                    "total_claims",
                    len(as_list(claims.get("claims")))
                )
            )

            claim_rows = get_result_rows(
                claims.get("claims")
            )

            if claim_rows:

                st.dataframe(
                    pd.DataFrame(claim_rows),
                    use_container_width=True,
                    hide_index=True
                )

            inconsistencies = get_result_rows(
                claims.get("potential_inconsistencies")
            )

            if inconsistencies:

                st.warning(
                    "Potential inconsistencies require human review. "
                    "They are not proof of deception."
                )

                st.dataframe(
                    pd.DataFrame(inconsistencies),
                    use_container_width=True,
                    hide_index=True
                )

            elif not claims:

                st.info("Claim consistency results are unavailable.")


        # Escalation
        with st.expander("Escalation assessment"):

            score = escalation.get(
                "escalation_score",
                escalation.get("score", 0)
            )

            a, b, c = st.columns(3)

            a.metric("Escalation score", str(score))

            b.metric(
                "Priority",
                clean_label(escalation.get("priority", "Unknown"))
            )

            c.metric(
                "Human review",
                "Required"
                if escalation.get("human_review_required")
                else "Not indicated"
            )

            if escalation.get("resolution_adjustment"):

                st.info(
                    escalation["resolution_adjustment"]
                )

            reasons = as_list(
                escalation.get(
                    "reasons",
                    escalation.get("reason", [])
                )
            )

            if reasons:

                st.markdown("**Reasons for escalation**")

                for reason in reasons:
                    st.write(f"• {reason}")


        # Routing
        with st.expander("Agent routing recommendation"):

            if routing:

                routing_display = {
                    "Recommended department": get_value(
                        routing,
                        ["recommended_department", "department"]
                    ),
                    "Recommended agent level": get_value(
                        routing,
                        ["recommended_agent_level", "agent_level"]
                    ),
                    "Routing priority": get_value(
                        routing,
                        ["routing_priority"]
                    ),
                    "Routing status": get_value(
                        routing,
                        ["routing_status"],
                        default="Recommendation only"
                    )
                }

                st.table(
                    pd.DataFrame(
                        list(routing_display.items()),
                        columns=["Field", "Recommendation"]
                    )
                )

                st.warning(
                    "Recommendation only — no agent assignment or transfer has occurred."
                )

            else:

                st.info("Routing recommendation is unavailable.")


        # Contact statistics
        with st.expander("Conversation and contact statistics"):

            if contact_stats:

                st.table(
                    pd.DataFrame(
                        list(contact_stats.items()),
                        columns=["Measure", "Value"]
                    )
                )

            else:

                st.info("No contact statistics were returned.")


        # System diagnostics
        with st.expander("System diagnostics"):

            st.write(
                f"**Main model:** "
                f"{'Loaded' if not analysis['model_error'] else 'Unavailable'}"
            )

            st.write(
                f"**Optional modules loaded:** "
                f"{len(MODULES)} / {len(MODULE_NAMES)}"
            )

            if analysis["model_error"]:
                st.error(analysis["model_error"])

            if MODULE_ERRORS:

                st.warning("Some optional modules could not be imported.")

                for name, error in MODULE_ERRORS.items():

                    st.write(f"**{name}:** {error}")

            else:

                st.success("No optional module import errors were recorded.")


        # Existing conversation history feature
        with st.expander("Anonymous conversation history"):

            st.caption(
                "This interface does not request a customer ID. "
                "Saved demo conversations use an anonymous internal key."
            )

            if st.button("Save this conversation", key="advanced_save"):

                saved, message = save_history(
                    conversation_id,
                    customer_id,
                    messages
                )

                if saved:
                    st.success(message)
                else:
                    st.error(message)

            history = load_history(customer_id)

            if not history.empty:

                st.dataframe(
                    history,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("No saved anonymous conversations were found.")


    # ============================================================
    # PAGE 3 — REPORTS AND DOWNLOADS
    # ============================================================

    elif selected_page == "Reports & Downloads":

        st.markdown("### Case reports and downloads")

        st.caption(
            "Export the conversation and its analysis for documentation, "
            "review and project demonstration."
        )

        st.markdown("#### Case report")

        report_lines = [
            "CUSTOMER SUPPORT CASE INTELLIGENCE REPORT",
            "=" * 55,
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "CASE OVERVIEW",
            f"Primary customer intent: {primary_message_intent}",
            f"Latest message intent: {latest_message_intent}",
            f"Overall model prediction: {analysis['model_intent']}",
            f"Overall model confidence: {analysis['model_confidence']:.4f}",
            f"Intent correction method: {analysis['correction_reason']}",
            f"Correction method: {analysis['correction_reason']}",
            f"Customer messages: {analysis['customer_count']}",
            f"Agent messages: {analysis['agent_count']}",
            "",
            "CUSTOMER INTELLIGENCE",
            f"Risk score: {risk['score']}/100",
            f"Risk level: {risk['level']}",
            f"Latest emotion: {latest_emotion}",
            f"Emotion progression: {progression.get('progression', 'Not available')}",
            "",
            "RESOLUTION",
            f"Status: {resolution['status']}",
            f"Next action: {resolution['next_action']}",
            *[f"Evidence: {e}" for e in resolution.get("evidence", [])],
            "",
            "ESCALATION",
            f"Score: {escalation.get('escalation_score', escalation.get('score', 'Not available'))}",
            f"Priority: {escalation.get('priority', 'Not available')}",
            f"Reasons: {escalation.get('reasons', [])}",
            "",
            "ROUTING RECOMMENDATION — NOT AN ASSIGNMENT",
            f"Department: {routing.get('recommended_department', routing.get('department', 'Not available'))}",
            f"Agent level: {routing.get('recommended_agent_level', routing.get('agent_level', 'Not available'))}",
            f"Priority: {routing.get('routing_priority', 'Not available')}",
            "",
            "CLAIM CONSISTENCY",
            str(claims),
            "",
            "EMOTION RESULTS",
            str(results.get("emotion_results")),
            "",
            "MESSAGE-LEVEL INTENTS",
            str(results.get("intent_results")),
            "",
            "FULL CONVERSATION",
            flatten_text(messages)
        ]

        report = "\n".join(report_lines)

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "↓ Download case report (.txt)",
                data=report.encode("utf-8"),
                file_name="customer_support_case_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col2:

            csv_data = pd.DataFrame(messages).to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "↓ Download conversation data (.csv)",
                data=csv_data,
                file_name="conversation_messages.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.divider()

        st.markdown("#### Report preview")

        with st.expander("View report content"):

            st.text(report)

else:

    if selected_page != "Case Dashboard":

        st.info(
            "No analyzed conversation is available yet. "
            "Open Case Dashboard, paste a conversation and analyze it first."
        )

    else:

        st.info(
            "Your dashboard is ready. Paste a customer-support conversation "
            "above and select Analyze Conversation to begin."
        )


# ------------------------- Footer -------------------------

st.divider()

st.caption(
    "Customer Support Case Intelligence · "
    "AI-assisted decision support · "
    "Emotion and risk indicators should be reviewed before customer-impacting action."
)