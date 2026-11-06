
import re
from collections import Counter


EMOTION_KEYWORDS = {
    "anger": [
        "angry", "furious", "outraged", "ridiculous",
        "unacceptable", "disgusted", "hate this",
        "fed up", "terrible service"
    ],

    "frustration": [
        "frustrated", "frustrating", "frustration",
        "annoyed", "irritated", "impatient",
        "tired of waiting", "still waiting",
        "waiting for days", "no response",
        "nobody is helping", "not helping",
        "as soon as possible", "again and again",
        "how much longer", "money back"
    ],

    "sadness": [
        "disappointed", "unhappy", "upset",
        "sad", "regret", "poor experience"
    ],

    "satisfaction": [
        "thank you", "thanks", "great",
        "excellent", "happy", "satisfied",
        "everything is fine", "problem solved",
        "resolved", "appreciate your help"
    ],

    "urgency": [
        "urgent", "immediately", "right now",
        "as soon as possible", "quickly",
        "emergency", "without delay"
    ]
}


def analyze_emotion(message):
    if not isinstance(message, str) or not message.strip():
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "evidence": []
        }

    text = message.lower()
    detected = {}

    for emotion, keywords in EMOTION_KEYWORDS.items():
        matches = [
            word for word in keywords
            if re.search(r"\b" + re.escape(word) + r"\b", text)
        ]

        if matches:
            detected[emotion] = matches

    if "anger" in detected:
        emotion = "anger"
        evidence = detected["anger"]
        confidence = 0.90

    elif "frustration" in detected:
        emotion = "frustration"
        evidence = detected["frustration"]
        confidence = 0.90

    elif "sadness" in detected:
        emotion = "sadness"
        evidence = detected["sadness"]
        confidence = 0.85

    elif "satisfaction" in detected:
        emotion = "satisfaction"
        evidence = detected["satisfaction"]
        confidence = 0.85

    else:
        emotion = "neutral"
        evidence = []
        confidence = 0.50

    return {
        "emotion": emotion,
        "confidence": confidence,
        "evidence": evidence
    }


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
            "evidence": result["evidence"]
        })

    return results


# Compatibility function expected by the existing project
def get_emotion_summary(emotion_results):

    if not emotion_results:
        return {
            "overall_emotion": "neutral",
            "total_messages": 0,
            "emotion_distribution": {}
        }

    emotions = [
        item.get("emotion", "neutral")
        for item in emotion_results
    ]

    distribution = dict(Counter(emotions))

    overall_emotion = max(
        distribution,
        key=distribution.get
    )

    return {
        "overall_emotion": overall_emotion,
        "total_messages": len(emotion_results),
        "emotion_distribution": distribution
    }
