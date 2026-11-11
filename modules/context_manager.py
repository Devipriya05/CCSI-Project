import re
from datetime import datetime

from modules.text_utils import normalize_text

# Kept for backward compatibility with older imports/tests.
SPEAKER_PATTERN = re.compile(r"^(Customer|Agent)\s*:\s*(.*)$", re.IGNORECASE)

_CUSTOMER_LABELS = {
    "customer", "client", "user", "cx", "caller", "buyer", "shopper", "guest", "visitor", "me", "consumer",
}
_AGENT_LABELS = {
    "agent", "support", "rep", "representative", "assistant", "bot", "staff", "csr", "advisor",
    "adviser", "operator", "helpdesk", "ai", "system", "specialist", "associate",
    "customer support", "customer service", "support agent", "service agent", "customer care",
}
# [10:32], (10:32 AM), 10:32:05 -, 2026-11-11 10:32 ...
_PREFIX = re.compile(
    r"^\s*(?:\[[^\]]{1,30}\]|\(?\d{1,4}[-/:]\d{1,2}(?:[-/:]\d{1,4})?(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?\s*(?:[AaPp][Mm])?\)?)\s*[-–]?\s*"
)
_LABELLED = re.compile(r"^\s*(?P<label>[A-Za-z][A-Za-z .'()-]{0,35}?)\s*[:：]\s*(?P<msg>.*)$")


def _speaker_from_alias(raw):
    raw = re.sub(r"[^a-z ]", " ", raw.lower())
    words = raw.split()
    if not words:
        return None
    joined = " ".join(words)
    if joined in _AGENT_LABELS or " ".join(words[:2]) in _AGENT_LABELS or words[0] in _AGENT_LABELS:
        return "Agent"        # "Agent Sarah", "Customer Support", "Support Team"
    if joined in _CUSTOMER_LABELS or words[0] in _CUSTOMER_LABELS:
        return "Customer"     # "Customer John"
    return None


def _speaker_from_label(label):
    # "Sarah (Agent)" -> role in brackets wins; otherwise use the label itself.
    for inner in re.findall(r"\((.*?)\)", label):
        found = _speaker_from_alias(inner)
        if found:
            return found
    return _speaker_from_alias(re.sub(r"\(.*?\)", " ", label))


def parse_conversation(conversation_text, conversation_id="CONV001"):
    """Parse a pasted chat into messages.

    Understands 'Customer:'/'Agent:' plus aliases (User, Support, Rep, Bot ...),
    timestamps in front of the label, and multi-line messages (unlabelled lines
    continue the previous message).  If *no* line has a speaker label, every
    line is treated as a customer message.
    """
    lines = [ln for ln in normalize_text_lines(conversation_text)]
    labelled = []  # (speaker or None, text) per line
    for line in lines:
        stripped = _PREFIX.sub("", line, count=1).strip() or line.strip()
        m = _LABELLED.match(stripped)
        speaker = _speaker_from_label(m.group("label")) if m else None
        labelled.append((speaker, m.group("msg").strip() if speaker else line.strip()))

    has_labels = any(sp for sp, _ in labelled)
    parsed = []  # [speaker or None, text]
    for speaker, text in labelled:
        if speaker or not has_labels:
            parsed.append([speaker, text])            # each line is its own message
        elif parsed:
            parsed[-1][1] = (parsed[-1][1] + " " + text).strip()   # continuation line
        else:
            parsed.append([None, text])               # stray text before first label

    messages = []
    for speaker, text in parsed:
        if not text:
            continue
        if speaker is None:
            if has_labels:
                continue  # stray text before the first label
            speaker = "Customer"
        sequence = len(messages) + 1
        messages.append({
            "conversation_id": conversation_id,
            "message_id": f"{conversation_id}_MSG{sequence:03d}",
            "sequence_number": sequence,
            "speaker": speaker,
            "message": text,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
    return messages


def normalize_text_lines(text):
    for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = normalize_text(line)
        if line:
            yield line


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


# --------------------------------------------------------------------------
# Resolution detection
# --------------------------------------------------------------------------
_NEGATOR = re.compile(r"\b(?:not|no|never|cannot|without|hardly|barely|nothing)\b|n't\b")

_UNRESOLVED = [
    r"\bstill (?:waiting|no\b|nothing|not\b|haven't|have not|hasn't|has not|didn't|did not|doesn't|does not|isn't|is not|can't|cannot|pending|unresolved|broken|stuck|missing|charged|an issue|a problem)",
    r"\bnot (?:yet |been )?(?:resolved|fixed|working|received|arrived|delivered|credited|refunded|solved|sorted|processed)\b",
    r"\b(?:haven't|have not|hasn't|has not|didn't|did not|never) (?:received|got|gotten|seen|heard|arrived|come|came|been (?:refunded|credited|delivered|resolved))\b",
    r"\b(?:isn't|is not|doesn't|does not|won't|will not|can't|cannot) (?:work|working|load|open|go through|resolved|fixed)\b",
    r"\bno (?:response|reply|answer|update|updates|resolution|progress|refund)\b",
    r"\b(?:nothing|nobody|no ?one)\b.{0,20}\b(?:happened|changed|help\w*|respon\w+|repl(?:y|ies|ied|ying)|answer\w*|call\w*|contact\w*|getting back)\b",
    r"\b(?:it(?:'s| has| is)? been|been waiting(?: for)?) (?:over |almost |nearly |more than |about )?(?:\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|several|many|few) (?:days?|weeks?|months?)\b",
    r"\b(?:problem|issue|error) (?:persists|remains|continues|is still)\b|\bsame (?:problem|issue|error)\b|\bremains? unresolved\b",
    r"\byet to (?:receive|get|hear)\b|\bnot heard (?:back|anything)\b",
    r"\bi(?:'m| am| have been|'ve been) waiting\b|\bbeen waiting\b",
    r"\b(?:it|this|that) (?:keeps|keep) (?:failing|crashing|happening)\b",
]
_RESOLVED = [
    r"\beverything(?:'s| is) (?:correct|okay|ok|fine|good|sorted|working)(?: now)?\b",
    r"\b(?:issue|problem|it|that|this) (?:is |was |has been |got )?(?:now )?(?:resolved|fixed|solved|sorted)\b",
    r"\b(?:that|this) (?:solves|solved|fixes|fixed|helps|helped|worked|works|did it|does it)\b",
    r"\b(?:works|working|worked) (?:now|fine|perfectly|well|great)\b",
    r"\bi (?:have |'ve |just |now )?(?:received|got|gotten|found) (?:it|my|the|them|that|everything)\b",
    r"\b(?:it|package|parcel|order|delivery|item|refund|money|payment) (?:has |have |just |finally )*(?:arrived|came|come|reached|credited|refunded|received|delivered)\b",
    r"\b(?:refund|money) (?:is |was |has been |has just been |just )(?:credited|received|refunded|back)\b|\b(?:refund|money) (?:credited|refunded)\b",
    r"\bno (?:further|more|other) (?:help|assistance|questions|issues|problems)\b(?: needed| required)?",
    r"\ball (?:good|set|sorted)\b",
    r"\bnever ?mind\b.{0,25}\b(?:fine|good|found|sorted|solved)\b|\bfound it\b",
    r"\bonly one (?:successful )?(?:charge|payment|transaction)\b",
    r"\bno duplicate (?:charge|payment|transaction)\b|\bthere was no duplicate\b|\bnot charged twice\b",
    r"\b(?:that'?s|that is|that will be|this is) (?:all|everything)\b|\bnothing else\b|\ball done\b",
    r"\b(?:thanks|thank you).{0,20}\b(?:solved|fixed|resolved|sorted|perfect|that helps)\b",
]
_WITHDRAWAL_ACK = [
    r"\bsorry for the confusion\b", r"\bmy (?:mistake|bad|error)\b", r"\bi was (?:wrong|mistaken)\b",
    r"\bapologi[sz]e for the confusion\b", r"\beverything is correct\b",
]
_WITHDRAWAL_VERIFY = [
    r"\bonly one (?:successful )?(?:charge|payment|transaction)\b", r"\bno duplicate (?:charge|payment|transaction)\b",
    r"\bi checked (?:my )?(?:bank statement|account|payment details)\b", r"\bnot charged twice\b",
]
_UNRESOLVED_RE = [re.compile(p, re.I) for p in _UNRESOLVED]
_RESOLVED_RE = [re.compile(p, re.I) for p in _RESOLVED]
_ACK_RE = [re.compile(p, re.I) for p in _WITHDRAWAL_ACK]
_VERIFY_RE = [re.compile(p, re.I) for p in _WITHDRAWAL_VERIFY]


def _first_match(regexes, text, guard_negation=False):
    for rx in regexes:
        for m in rx.finditer(text):
            if guard_negation:
                before = re.findall(r"[a-z']+", text[:m.start()])[-3:]
                if _NEGATOR.search(" ".join(before)):
                    continue
            return m.group(0)
    return None


def detect_resolution_status(messages):
    """Conservative, rule-based case-state detection; not proof of backend resolution.

    The latest customer message decides.  Negation is respected ("not working
    now" is unresolved) and so are closing phrases ("that's all, thanks").
    """
    customer = [m for m in messages if m.get("speaker", "").lower() == "customer"]
    if not customer:
        return {"status": "Unknown", "resolved": False, "evidence": "No customer messages",
                "confidence": 0.0,
                "next_action": "Review the conversation."}

    latest = customer[-1].get("message", "").strip()
    text = normalize_text(latest).lower()

    hit = _first_match(_UNRESOLVED_RE, text)
    if hit:
        return {"status": "Unresolved", "resolved": False, "evidence": latest, "confidence": 0.9,
                "next_action": "Continue investigating the outstanding issue and provide an update."}

    # "No, that's all, thanks" / "Yes, that fixed it": drop the answer word so it
    # is not mistaken for a negation of what follows.
    text = re.sub(r"^(?:no|nope|yes|yeah|yep|ok|okay|sure)\b[\s,.!-]*", "", text)
    hit = _first_match(_RESOLVED_RE, text, guard_negation=True)
    if hit:
        return {"status": "Resolved", "resolved": True, "evidence": latest, "confidence": 0.9,
                "next_action": "Confirm whether further assistance is needed; if no additional issue remains, close the case."}

    # Complaint withdrawn earlier in the chat: verification + acknowledgement.
    all_text = " ".join(normalize_text(m.get("message", "")).lower() for m in customer)
    if _first_match(_VERIFY_RE, all_text) and _first_match(_ACK_RE, all_text):
        return {"status": "Resolved", "resolved": True,
                "evidence": "Customer verified the facts and acknowledged the earlier complaint was a mistake.",
                "confidence": 0.85,
                "next_action": "Mark the complaint as withdrawn and close the case if no other issue remains."}

    return {"status": "Not confirmed", "resolved": False, "evidence": latest, "confidence": 0.5,
            "next_action": "Follow up with the customer to confirm whether the issue is resolved."}
