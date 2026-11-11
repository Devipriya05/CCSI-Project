"""Customer emotion analysis for support conversations.

Rule-based and fully transparent (no extra dependencies), but built to cope
with how people really write:

* stems and many synonyms, not a handful of exact keywords
  ("annoyed", "annoying", "so annoyed", "super irritated" ...)
* negation          - "not happy" is dissatisfaction, "not angry" is not anger
* sarcasm           - "thanks for nothing" is anger, not gratitude
* implicit emotion  - "nobody is answering", "still waiting", "for the third time"
* intensity         - "very", "extremely", CAPS, "!!!"
* typos             - "anooyed", "frustated", "dissapointed"
* mixed messages    - "thanks, but it's still broken"

Labels: anger, frustration, anxiety, sadness, confusion, satisfaction,
urgency (only when nothing else applies) and neutral.
"""
import re
from collections import Counter
from difflib import get_close_matches

from modules.text_utils import collapse_repeats, normalize_text, strip_neutral_idioms

# Higher = worse for the customer relationship.
EMOTION_SEVERITY = {
    "satisfaction": 0,
    "neutral": 1,
    "confusion": 2,
    "urgency": 1,   # not an emotion: must not look like "deterioration" after a neutral message
    "anxiety": 3,
    "sadness": 3,
    "frustration": 4,
    "anger": 5,
}

NEGATIVE_EMOTIONS = ("anger", "frustration", "anxiety", "sadness", "confusion")
# Tie-break order when scores are equal.
_PRIORITY = ("anger", "frustration", "anxiety", "sadness", "confusion", "satisfaction")
THRESHOLD = 0.85  # minimum score for a message to receive an emotion label

# --------------------------------------------------------------------------
# Lexicon: (emotion, weight, negation_applies, [regex, ...])
# negation_applies=False for phrases that already contain their own negation.
# --------------------------------------------------------------------------
_LEXICON_SPEC = [
    # ----------------------------- ANGER ---------------------------------
    ("anger", 2.2, True, [
        r"\bfurious\b", r"\boutrag\w+", r"\blivid\b", r"\benraged?\b", r"\binfuriat\w+",
        r"\bdisgust\w*", r"\bangry\b", r"\banger\b", r"\bpissed\b", r"\bfuming\b",
        r"\bappall\w+", r"\bscam(?:med|mers?)?\b", r"\bfraud\w*", r"\bthie(?:f|ves)\b",
        r"\brip(?:ped|ping)? (?:me |us )?off\b", r"\bstole\b|\bstealing\b",
    ]),
    ("anger", 1.8, True, [
        r"\bunacceptable\b", r"\bridiculous\b", r"\babsurd\b", r"\bpathetic\b",
        r"\bdisgraceful\b", r"\bshameful\b", r"\bhate\b", r"\bworst\b", r"\bhorrible\b",
        r"\bterrible\b", r"\bawful\b", r"\bdreadful\b", r"\bincompeten\w+", r"\buseless\b",
        r"\bgarbage\b", r"\btrash\b", r"\bunprofessional\b", r"\brude\b", r"\bdisrespect\w*",
        r"\bdamn\b", r"\bwtf\b", r"\bcrap\b", r"\bbullshit\b", r"\bnonsense\b", r"\bsucks?\b",
        r"\b(?:is|was|what) an? (?:total |complete )?(?:joke|disgrace|nightmare|mess)\b",
        r"\bmad\b",
    ]),
    ("anger", 2.0, False, [
        r"\bnever\b.{0,30}\b(?:again|return|come back|recommend)\b",
        r"\bnever (?:buy|order|shop|use|purchase)\w*\b",
        r"\b(?:lawyer|attorney|legal action|sue (?:you|them|the company)|consumer (?:court|forum|protection)|ombudsman)\b",
        r"\breport (?:you|this|it|the company|your company)\b",
        r"\bchargeback\b|\bdispute (?:the |this )?charge\b",
        r"\b(?:worst|terrible|awful|horrible|poor|bad|pathetic|appalling) (?:customer )?(?:service|support|experience|company)\b",
    ]),

    # --------------------------- FRUSTRATION -----------------------------
    ("frustration", 1.8, True, [
        r"\bfrustrat\w+", r"\bannoy\w*", r"\birritat\w+", r"\baggravat\w+",
        r"\bexasperat\w+", r"\bimpatien\w+", r"\bhassle\b", r"\bnuisance\b",
    ]),
    ("frustration", 2.0, False, [
        r"\bfed up\b", r"\bsick of\b", r"\btired of\b", r"\bhad enough\b", r"\bsick and tired\b",
        r"\b(?:nobody|no ?one)\b.{0,25}\b(?:help\w*|respon\w+|answer\w*|repl(?:y|ies|ied|ying)|getting back|cares?|listen\w*|bothered|doing anything|call\w*|contact\w*|reach\w*|pick\w* up)\b",
        r"\bwaste(?:d)? (?:of )?(?:my |our |so much |a lot of )?(?:time|money|energy)\b|\bwasting (?:my )?time\b",
        r"\brun ?around\b|\bbounced? (?:me )?around\b|\bpass(?:ed|ing) (?:me )?(?:around|from)\b",
    ]),
    ("frustration", 1.8, False, [
        r"\b(?:for|over|past) the (?:second|third|fourth|fifth|sixth|umpteenth|\d+(?:st|nd|rd|th)) time\b",
        r"\bagain and again\b|\bover and over\b|\btime and again\b|\bonce again\b|\byet again\b",
        r"\b(?:not|isn't|aren't|wasn't|weren't|n't) (?:being )?(?:helping|helpful|useful|responding|answering|listening|solving|fixing)\b|\bunhelpful\b",
        r"\bignor(?:e|ed|ing)\b",
        r"\btaking (?:forever|so long|too long|ages)\b|\bforever to\b",
        r"\b(?:not|isn't|is not) (?:acceptable|good enough|okay|ok)\b",
        r"\bunbelievable\b|\bcan't believe\b|\bcannot believe\b|\bgive up\b|\bgiving up\b",
    ]),
    ("frustration", 1.6, False, [
        r"\b(?:speak|talk) (?:to|with) (?:a |an |your |the |some )?(?:manager|supervisor|senior|someone (?:else|in charge|higher))\b",
        r"\bescalate\b|\bget me (?:a |your )?(?:manager|supervisor)\b|\bhigher (?:up|authority)\b",
        r"\b(?:tried|called|contacted|emailed|messaged|asked|chatted|written|reached out)\b.{0,30}\b(?:many|several|multiple|numerous|countless|a few|three|four|five|six|\d+) times\b",
        r"\b(?:tried|called|contacted|emailed|messaged) (?:you |support |them )?(?:twice|again)\b",
        r"\bkeeps? (?:crashing|failing|freezing|disconnecting)\b",
    ]),
    ("frustration", 1.4, False, [
        r"\bkeeps? (?:calling|asking|getting|sending|telling|having|trying|waiting)\b",
        r"\bstill (?:waiting|no\b|nothing|not\b|haven't|have not|hasn't|has not|didn't|did not|doesn't|does not|isn't|is not|can't|cannot|pending|unresolved|broken|stuck)",
        r"\bno (?:response|reply|answer|update|updates|resolution|progress|follow[- ]?up|feedback)\b",
        r"\b(?:doesn't|does not|didn't|did not|isn't|is not|won't|will not|can't|cannot|not) (?:work|working|load|loading|open|go through|going through)\b",
    ]),
    ("frustration", 1.2, False, [
        r"\bhow much longer\b",
        r"\b(?:why|how) (?:is|has|does) (?:it|this|that) (?:taking|take|taken) (?:so )?long\b",
        r"\bno help\b",
    ]),
    ("frustration", 1.0, False, [
        r"\b(?:waiting|waited)\b (?:for )?(?:over |almost |nearly |more than |already )?(?:\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|several|many|few) (?:days?|weeks?|months?|hours?)\b",
        r"\bmoney back\b",
        r"\bwhatever\b",
        r"\b(?:it(?:'s| has| is)? been|it's been|been waiting(?: for)?) (?:over |almost |nearly |more than |about )?(?:\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|several|many|few) (?:days?|weeks?|months?)\b",
    ]),
    ("frustration", 0.9, False, [
        r"\bhow long (?:will|is|does) (?:this|it) (?:going to )?(?:take|last)\b",
        r"\bseriously\b",
    ]),
    ("frustration", 0.6, False, [  # factual problem statements: sub-threshold on their own
        r"\b(?:haven't|have not|hasn't|has not|didn't|did not|never) (?:received|got|gotten|arrived)\b",
        r"\bnot (?:received|arrived|delivered|working)\b",
        r"\b(?:failed|failing|failure|error|broken|crashed)\b",
        r"\bstill\b",
    ]),

    # ------------------------------ SADNESS ------------------------------
    ("sadness", 1.8, True, [
        r"\bdisappoint\w*", r"\bunhappy\b", r"\bupset\b", r"\bdissatisf\w+", r"\bunsatisf\w+",
        r"\bsadden\w*|\bsad\b", r"\bheartbr\w+", r"\bregret\w*", r"\bdiscourag\w+",
    ]),
    ("sadness", 1.8, False, [
        r"\blet (?:me|us|you) down\b|\bletting me down\b",
        r"\bexpected (?:better|more)\b", r"\bnot what i (?:expected|ordered|wanted|paid)\b",
        r"\bnot (?:impressed|thrilled)\b",
        r"\b(?:poor|bad|low) (?:quality|experience)\b",
    ]),

    # ------------------------------ ANXIETY ------------------------------
    ("anxiety", 1.6, True, [
        r"\bworr(?:y|ied|ies|ying)\b", r"\banxi(?:ous|ety)\b", r"\bnervous\b", r"\bscared\b",
        r"\bafraid\b", r"\bpanic\w*", r"\bstress\w*", r"\bconcerned\b", r"\bterrified\b",
        r"\bfrightened\b", r"\bdesperate\w*", r"\bfear\w*",
    ]),
    ("anxiety", 1.6, False, [
        r"\bis my (?:money|payment|account|data|card|order|refund) (?:safe|secure|gone|lost)\b",
        r"\b(?:lose|lost|losing) (?:my )?(?:money|account|data|job)\b",
        r"\b(?:don't|do not) know what to do\b|\bat a loss\b",
    ]),
    ("anxiety", 0.9, False, [
        r"\bwhat if\b",
        r"\bmy (?:rent|bills|salary|wedding|flight|exam|interview|surgery|medication|deadline)\b",
    ]),
    ("anxiety", 0.7, False, [
        r"\bplease help\b|\bi need your help\b|\bhelp me\b",
        r"\breally need (?:this|it|my)\b",
        r"\bhope (?:you|this|it|we)\b",
    ]),

    # ----------------------------- CONFUSION -----------------------------
    ("confusion", 1.4, True, [
        r"\bconfus\w+",
        r"\bunclear\b",
        r"\bno idea\b",
    ]),
    ("confusion", 1.4, False, [
        r"\b(?:don't|do not|doesn't|does not|didn't|did not|can't|cannot|couldn't) (?:really )?(?:understand|get (?:it|this|why)|figure (?:it |this )?out|see why|know why)\b",
        r"\b(?:makes?|made) no sense\b|\b(?:doesn't|does not) make sense\b|\bnot (?:very |really |quite )?(?:clear|sure)\b",
        r"\b(?:i'm|i am|feel|totally|completely) lost\b",
    ]),
    ("confusion", 1.0, False, [
        r"\bwhat (?:does|do) (?:this|that|it|you) mean\b",
        r"\bwhy (?:was|am|did|were) (?:i|my \w+|there|it) (?:charged|billed|being charged|showing|deducted|debited|cancelled|canceled)\b",
    ]),

    # --------------------------- SATISFACTION ----------------------------
    ("satisfaction", 1.8, True, [
        r"\bhappy\b", r"\bglad\b", r"\bpleased\b", r"\bsatisfied\b", r"\bdelighted\b",
        r"\bthrilled\b", r"\brelieved\b", r"\bimpressed\b", r"\bgrateful\b", r"\bthankful\b",
        r"\byou(?:'re| are) (?:the best|awesome|great|amazing|a lifesaver)\b|\blifesaver\b",
    ]),
    ("satisfaction", 1.5, True, [
        r"\bappreciat\w+",
        r"\b(?:works|working|worked) (?:now|perfectly|fine|well|great)\b",
        r"\b(?:problem|issue|it|that) (?:is |was |has been |got )?(?:now )?(?:resolved|solved|fixed|sorted)\b",
        r"\beverything(?:'s| is) (?:fine|good|great|sorted|working)\b",
        r"\bthat (?:helps|helped|worked|did it|solved)\b",
        r"\blove (?:it|this|the|your)\b",
    ]),
    ("satisfaction", 1.4, True, [
        r"\b(?:great|excellent|awesome|wonderful|fantastic|amazing|perfect|brilliant|superb|outstanding|lovely)\b",
    ]),
    ("satisfaction", 1.2, True, [
        r"\bthank(?:s| you| u)\b|\bthanx\b|\bthx\b|\bcheers\b",
        r"\bhelpful\b", r"\bgood job\b|\bwell done\b|\bkudos\b",
        r"\b(?:all|everything)(?: is)? good\b|\ball set\b",
    ]),
]

# Positive words that, when negated, mean "this is NOT working for me".
_NEGATED_POSITIVE_TO_FRUSTRATION = {"helpful", "working", "worked", "works", "fixed", "resolved", "solved", "helps", "helped"}

# Explicit sarcasm: anger/frustration, and the clause's positive words are ignored.
_SARCASM = [
    r"\bthank(?:s| you)(?: so much| very much)?,? for nothing\b",
    r"\bthanks? for (?:nothing|ruining|wasting|making this worse)\b",
    r"\b(?:oh )?(?:great|wonderful|fantastic|perfect),? (?:just|now|another|more)\b",
    r"\bjust (?:great|perfect|wonderful|fantastic)\b",
    r"\byeah,? right\b",
    r"\b(?:real|very|so) helpful\b\.{3}",
]

_URGENCY = [
    r"\burgent\w*", r"\basap\b", r"\bimmediate\w*", r"\bright now\b", r"\bright away\b",
    r"\bas soon as possible\b", r"\bemergency\b", r"\bwithout delay\b", r"\btime[- ]sensitive\b",
    r"\bneed (?:it|this|that) (?:now|today|tonight|tomorrow)\b", r"\bdeadline\b",
    r"\b(?:by|before|until) (?:today|tonight|tomorrow|noon|midnight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
]

_NEGATORS = re.compile(r"\b(?:not|no|never|cannot|without|hardly|barely|neither|nor|nothing)\b|n't\b")
_INTENSIFIERS = re.compile(r"\b(?:very|really|so+|extremely|totally|absolutely|completely|utterly|incredibly|super|truly|deeply|highly|damn|seriously)\b")
_DIMINISHERS = re.compile(r"\b(?:a bit|a little|slightly|somewhat|kind of|kinda|sort of|bit)\b")

_LEXICON = [
    (emotion, weight, neg_ok, [re.compile(p, re.I) for p in patterns])
    for emotion, weight, neg_ok, patterns in _LEXICON_SPEC
]
_SARCASM_RE = [re.compile(p, re.I) for p in _SARCASM]
_URGENCY_RE = [re.compile(p, re.I) for p in _URGENCY]

# Words we are willing to auto-correct typos towards (single words, >= 6 letters).
_TYPO_VOCAB = sorted({
    "annoyed", "annoying", "frustrated", "frustrating", "disappointed", "disappointing",
    "unacceptable", "ridiculous", "furious", "terrible", "horrible", "unhappy",
    "dissatisfied", "irritated", "irritating", "outraged", "worried", "confused",
    "confusing", "satisfied", "excellent", "wonderful", "appreciate", "appreciated",
    "unprofessional", "incompetent", "disgusting", "pathetic", "useless", "anxious",
    "stressed", "impatient", "disgraceful", "unbelievable", "dreadful", "fantastic",
})
_TYPO_SET = set(_TYPO_VOCAB)


def _fix_typos(text):
    def repl(match):
        word = match.group(0)
        if word in _TYPO_SET:
            return word
        for cand in get_close_matches(word, _TYPO_VOCAB, n=1, cutoff=0.84):
            # Guard against "acceptable"->"unacceptable", "satisfied"->"dissatisfied".
            if cand[0] == word[0] and abs(len(cand) - len(word)) <= 1:
                return cand
        return word
    return re.sub(r"[a-z']{6,}", repl, text)


def _split_clauses(text):
    """Split into clauses; later halves of a contrast ('..., but ...') are flagged."""
    clauses = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        parts = re.split(r"\s*,?\s*\b(?:but|however|although|though)\b\s*,?\s*", sentence)
        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                clauses.append((part, i > 0))
    return clauses


def _window_tokens(clause, start, n):
    return re.findall(r"[a-z']+", clause[:start])[-n:]


def _empty_result():
    return {
        "emotion": "neutral", "confidence": 0.5, "evidence": [],
        "intensity": "low", "secondary_emotions": [], "urgent": False,
    }


def analyze_emotion(message):
    """Return the dominant emotion of one customer message with evidence."""
    if not isinstance(message, str) or not message.strip():
        return _empty_result()

    raw = normalize_text(message)
    text = strip_neutral_idioms(collapse_repeats(raw.lower()))
    text = _fix_typos(text)

    scores = Counter()
    evidence = []
    urgent = False

    for clause, after_contrast in _split_clauses(text):
        boost = 1.2 if after_contrast else 1.0
        sarcastic = False

        for rx in _SARCASM_RE:
            for m in rx.finditer(clause):
                sarcastic = True
                scores["anger"] += 1.6 * boost
                scores["frustration"] += 1.0 * boost
                evidence.append(f"{m.group(0).strip()} (sarcasm)")

        for rx in _URGENCY_RE:
            for m in rx.finditer(clause):
                if not _NEGATORS.search(" ".join(_window_tokens(clause, m.start(), 2))):
                    urgent = True

        seen_spans = []
        for emotion, weight, neg_ok, regexes in _LEXICON:
            for rx in regexes:
                for m in rx.finditer(clause):
                    span = (m.start(), m.end())
                    # Count each stretch of text once, by its strongest entry.
                    if any(s <= span[0] and span[1] <= e for s, e in seen_spans):
                        continue
                    if emotion == "satisfaction" and sarcastic:
                        continue
                    w = weight * boost
                    window = _window_tokens(clause, m.start(), 3)
                    negated = neg_ok and bool(_NEGATORS.search(" ".join(window)))
                    label, note = emotion, m.group(0).strip()

                    if negated:
                        if emotion == "satisfaction":
                            words = set(re.findall(r"[a-z']+", m.group(0)))
                            label = "frustration" if words & _NEGATED_POSITIVE_TO_FRUSTRATION else "sadness"
                            w = max(w, 1.5)
                            note = f"not {m.group(0).strip()}"
                        else:
                            continue  # "not angry", "no need to worry" -> ignore
                    else:
                        near = " ".join(_window_tokens(clause, m.start(), 2))
                        if _INTENSIFIERS.search(near):
                            w *= 1.35
                        elif _DIMINISHERS.search(clause[max(0, m.start() - 12):m.start()]):
                            w *= 0.6
                    seen_spans.append(span)
                    scores[label] += w
                    evidence.append(note)

    # ---- message-level signals: shouting and repeated punctuation -------
    letters = [c for c in raw if c.isalpha()]
    shouting = len(letters) >= 8 and len(raw.split()) >= 2 and sum(c.isupper() for c in letters) / len(letters) >= 0.7
    heavy_punct = bool(re.search(r"[!?]{3,}|!{2,}", raw))
    neg_total = sum(scores[e] for e in NEGATIVE_EMOTIONS)
    if shouting:
        if neg_total:
            for e in NEGATIVE_EMOTIONS:
                scores[e] *= 1.3
        elif not scores["satisfaction"]:
            scores["frustration"] += 0.7
        evidence.append("ALL CAPS")
    if heavy_punct:
        if neg_total:
            for e in NEGATIVE_EMOTIONS:
                scores[e] *= 1.15
        elif scores["satisfaction"]:
            scores["satisfaction"] *= 1.15
        else:
            scores["frustration"] += 0.9
        evidence.append("emphatic punctuation")

    # ---- politeness opener: "Thanks, but ..." should not read as satisfied -
    neg_total = sum(scores[e] for e in NEGATIVE_EMOTIONS)
    if scores["satisfaction"] and neg_total >= THRESHOLD:
        scores["satisfaction"] *= 0.4

    # ---- choose the dominant emotion ------------------------------------
    ranked = sorted(
        ((e, s) for e, s in scores.items() if s >= THRESHOLD),
        key=lambda x: (-round(x[1], 2), _PRIORITY.index(x[0]) if x[0] in _PRIORITY else 99),
    )

    result = _empty_result()
    result["urgent"] = urgent
    if ranked:
        top, top_score = ranked[0]
        result["emotion"] = top
        result["confidence"] = round(min(0.97, 0.6 + 0.12 * top_score), 2)
        result["intensity"] = "high" if top_score >= 3.0 else "medium" if top_score >= 1.8 else "low"
        result["secondary_emotions"] = [e for e, _ in ranked[1:]]
        # evidence without the pseudo-signals if real words exist
        result["evidence"] = list(dict.fromkeys(evidence))
    elif urgent:
        result["emotion"] = "urgency"
        result["confidence"] = 0.7
        result["evidence"] = ["urgency wording"]
    else:
        result["confidence"] = 0.6
        result["evidence"] = list(dict.fromkeys(evidence))  # sub-threshold hints, if any
    return result


def detect_emotion(message):
    return analyze_emotion(message)


# Compatibility function expected by the existing project
def analyze_customer_emotions(messages):
    results = []
    for message in messages:
        if message.get("speaker", "").lower() != "customer":
            continue
        text = message.get("message", "")
        result = analyze_emotion(text)
        results.append({
            "message_id": message.get("message_id"),
            "sequence_number": message.get("sequence_number"),
            "message": text,
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "intensity": result["intensity"],
            "secondary_emotions": result["secondary_emotions"],
            "urgent": result["urgent"],
            "evidence": result["evidence"],
        })
    return results


# Compatibility function expected by the existing project
def get_emotion_summary(emotion_results):
    if not emotion_results:
        return {"overall_emotion": "neutral", "latest_emotion": "neutral", "peak_emotion": "neutral",
                "total_messages": 0, "emotion_distribution": {}, "urgent_messages": 0}

    ordered = sorted(emotion_results, key=lambda x: x.get("sequence_number") or 0)
    emotions = [item.get("emotion", "neutral") for item in ordered]
    distribution = dict(Counter(emotions))

    non_neutral = {e: c for e, c in distribution.items() if e != "neutral"}
    if non_neutral:
        overall = max(non_neutral, key=lambda e: (non_neutral[e] * (EMOTION_SEVERITY.get(e, 1) + 1), EMOTION_SEVERITY.get(e, 1)))
    else:
        overall = "neutral"
    peak = max(emotions, key=lambda e: EMOTION_SEVERITY.get(e, 1))

    return {
        "overall_emotion": overall,
        "latest_emotion": emotions[-1],
        "peak_emotion": peak,
        "total_messages": len(emotion_results),
        "emotion_distribution": distribution,
        "urgent_messages": sum(1 for item in ordered if item.get("urgent")),
    }
