import re
from datetime import datetime

SPEAKER_PATTERN = re.compile(r"^(Customer|Agent)\s*:\s*(.*)$", re.IGNORECASE)


def parse_conversation(conversation_text, conversation_id="CONV001"):
    """Parse lines formatted as 'Customer: ...' or 'Agent: ...'."""
    messages = []
    for line in (conversation_text or "").strip().splitlines():
        match = SPEAKER_PATTERN.match(line.strip())
        if not match:
            continue
        speaker, message_text = match.group(1).capitalize(), match.group(2).strip()
        if not message_text:
            continue
        sequence = len(messages) + 1
        messages.append({
            "conversation_id": conversation_id,
            "message_id": f"{conversation_id}_MSG{sequence:03d}",
            "sequence_number": sequence,
            "speaker": speaker,
            "message": message_text,
            "timestamp": datetime.now().isoformat(timespec="seconds")
        })
    return messages


def get_conversation_summary(messages):
    customer = [m for m in messages if m.get("speaker", "").lower() == "customer"]
    agents = [m for m in messages if m.get("speaker", "").lower() == "agent"]
    return {
        "conversation_id": messages[0].get("conversation_id") if messages else None,
        "total_messages": len(messages),
        "customer_messages": len(customer),
        "agent_messages": len(agents),
        "last_speaker": messages[-1].get("speaker") if messages else None,
        "conversation_text": " ".join(m.get("message", "") for m in messages)
    }


def detect_resolution_status(messages):
    """Conservative, rule-based case-state detection; not proof of backend resolution."""
    customer = [m for m in messages if m.get("speaker", "").lower() == "customer"]
    if not customer:
        return {"status": "Unknown", "resolved": False, "evidence": "No customer messages", "confidence": 0.0}

    latest = customer[-1].get("message", "").strip()
    text = latest.lower()
    unresolved = [
        r"\bstill (?:waiting|not|haven't|have not)\b", r"\bnot resolved\b",
        r"\bnot fixed\b", r"\bhasn't arrived\b", r"\bnot received\b",
        r"\bproblem persists\b", r"\bstill (?:doesn't|does not|isn't|is not) work\b"
    ]
    resolved = [
        r"\beverything is (?:correct|okay|ok|fine)\b", r"\bissue is resolved\b",
        r"\bproblem is resolved\b", r"\bthat (?:solves|solved) it\b",
        r"\bworking now\b", r"\bi received (?:it|the package|the refund)\b",
        r"\bonly one successful transaction\b", r"\bno duplicate (?:charge|transaction)\b",
        r"\bthere was only one (?:charge|transaction|payment)\b",
        r"\bnever mind\b.*\b(all good|it's fine|it is fine)\b"
    ]
    if any(re.search(p, text) for p in unresolved):
        return {"status": "Unresolved", "resolved": False, "evidence": latest, "confidence": 0.90}
    if any(re.search(p, text) for p in resolved):
        return {"status": "Resolved", "resolved": True, "evidence": latest, "confidence": 0.90}
    return {"status": "Not confirmed", "resolved": False, "evidence": latest, "confidence": 0.50}
