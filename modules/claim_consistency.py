import re

from modules.text_utils import normalize_text

# Claims are extracted as customer-stated evidence, not treated as verified facts.
CLAIM_PATTERNS = {
    "duplicate_charge": {
        "allegation": [r"\bcharged twice\b", r"\bdouble[- ]charged\b", r"\bduplicate charge\b", r"\bcharged more than once\b"],
        "verified_not_duplicate": [r"\bonly one successful transaction\b", r"\bonly one (?:successful )?(?:charge|transaction|payment)\b", r"\bno duplicate (?:charge|transaction|payment)\b", r"\bwas not charged twice\b"]
    },
    "payment": {
        "negative": [r"\bpayment failed\b", r"\bpayment not made\b", r"\bpayment was not successful\b", r"\btransaction failed\b"],
        "positive": [r"\bpayment successful\b", r"\bpayment completed\b", r"\bpayment was made\b", r"\bsuccessful transaction\b"]
    },
    "delivery": {
        "negative": [r"\bnot received\b", r"\b(?:haven't|have not|didn't|did not|never) (?:received|got|gotten)\b", r"\bnever arrived\b", r"\bnot (?:been )?delivered\b", r"\bstill waiting\b", r"\b(?:hasn't|has not) arrived\b"],
        "positive": [r"\bwas delivered\b", r"\bhas been delivered\b", r"\bi (?:received|got) (?:the |my )?(?:package|parcel|order|item)\b", r"\bpackage arrived\b", r"\border arrived\b", r"\bit arrived\b", r"\breceived the package\b"]
    },
    "refund": {
        "negative": [r"\brefund not received\b", r"\bhaven't received.*refund\b", r"\brefund is pending\b", r"\bwaiting for.*refund\b"],
        "positive": [r"\brefund received\b", r"\brefund credited\b", r"\bmoney refunded\b", r"\bgot my refund\b"]
    }
}


def identify_claim(message):
    text = normalize_text(message or "")
    for topic, categories in CLAIM_PATTERNS.items():
        for status, patterns in categories.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return {"topic": topic, "status": status, "evidence": text}
    return None


def analyze_claim_consistency(messages):
    claims = []
    for msg in messages:
        if msg.get("speaker", "").lower() != "customer":
            continue
        claim = identify_claim(msg.get("message", ""))
        if claim:
            claims.append({"message_id": msg.get("message_id"),
                           "sequence_number": msg.get("sequence_number"),
                           **claim})

    inconsistencies = []
    for i, first in enumerate(claims):
        for second in claims[i + 1:]:
            if first["topic"] != second["topic"] or first["status"] == second["status"]:
                continue
            pair = {first["status"], second["status"]}
            # A later verification/correction of a duplicate-charge concern is a normal case update.
            if first["topic"] == "duplicate_charge" and pair == {"allegation", "verified_not_duplicate"}:
                continue
            # A later explicit correction is not automatically suspicious.
            later = second["evidence"].lower()
            correction = any(x in later for x in ("actually", "to clarify", "i meant", "sorry for the confusion", "correction"))
            if correction:
                continue
            inconsistencies.append({
                "topic": first["topic"], "status_1": first["status"], "status_2": second["status"],
                "message_1": first["evidence"], "message_2": second["evidence"],
                "message_id_1": first["message_id"], "message_id_2": second["message_id"],
                "assessment": "Potential inconsistency - requires context and human verification",
                "recommendation": "Ask for clarification; do not infer deception from this signal alone"
            })
    return {"total_claims": len(claims), "claims": claims,
            "total_inconsistencies": len(inconsistencies), "inconsistencies": inconsistencies}
