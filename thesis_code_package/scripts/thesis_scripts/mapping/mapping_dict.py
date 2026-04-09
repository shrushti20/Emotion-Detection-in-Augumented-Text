# mapping_dict.py
# Mapping from GoEmotions 28-label space → CONTARGA 8-label space.
# Anything mapped to "neutral" is treated as "other / not in CONTARGA".

GOEMO_TO_CONTARGA = {
    # Anger cluster
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",

    # Disgust cluster
    "disgust": "disgust",

    # Fear cluster
    "fear": "fear",
    "nervousness": "fear",

    # Joy / positive affect cluster
    "joy": "joy",
    "amusement": "joy",
    "love": "joy",
    "optimism": "joy",
    "caring": "joy",
    "excitement": "joy",
    "admiration": "joy",
    "desire": "joy",

    # Pride / status
    "pride": "pride",
    "approval": "pride",

    # Relief /gratitude-ish
    "relief": "relief",
    "gratitude": "relief",

    # Sadness / loss / regret cluster
    "sadness": "sadness",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "embarrassment": "sadness",

    # Surprise / knowledge change cluster
    "surprise": "surprise",
    "curiosity": "surprise",
    "realization": "surprise",

    # Outside the 8 target argument emotions
    "confusion": "neutral",
    "neutral": "neutral",
}


#Now we have an explicit mapping for all 28 GoEmotions labels into:

#{anger, disgust, fear, joy, pride, relief, sadness, surprise, neutral