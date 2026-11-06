
EMOTION_SEVERITY = {
    "satisfaction": 0,
    "neutral": 1,
    "confusion": 2,
    "sadness": 3,
    "frustration": 4,
    "anger": 5
}


def analyze_emotion_progression(emotion_results):

    if not emotion_results:
        return {
            "progression": "Insufficient data",
            "first_emotion": None,
            "latest_emotion": None,
            "peak_emotion": None,
            "severity_change": 0,
            "message_timeline": []
        }

    # Maintain chronological message order
    ordered = sorted(
        emotion_results,
        key=lambda x: x.get("sequence_number", 0)
    )

    emotions = [
        item.get("emotion", "neutral").lower()
        for item in ordered
    ]

    scores = [
        EMOTION_SEVERITY.get(emotion, 1)
        for emotion in emotions
    ]

    first_emotion = emotions[0]
    latest_emotion = emotions[-1]

    first_score = scores[0]
    latest_score = scores[-1]

    # Find the highest emotional severity
    peak_index = scores.index(max(scores))
    peak_emotion = emotions[peak_index]
    peak_score = scores[peak_index]

    # Determine emotional progression
    if len(emotions) == 1:
        progression = "Insufficient data"

    elif peak_score > first_score and latest_score < peak_score:
        progression = "Deteriorated then improved"

    elif latest_score > first_score:
        progression = "Deteriorating"

    elif latest_score < first_score:
        progression = "Improving"

    else:
        progression = "Stable"

    # Build chronological emotion timeline
    timeline = []

    for item, emotion, score in zip(ordered, emotions, scores):

        timeline.append({
            "message_id": item.get("message_id"),
            "sequence_number": item.get("sequence_number"),
            "emotion": emotion,
            "severity": score
        })

    return {
        "progression": progression,
        "first_emotion": first_emotion,
        "latest_emotion": latest_emotion,
        "peak_emotion": peak_emotion,
        "severity_change": latest_score - first_score,
        "message_timeline": timeline
    }
