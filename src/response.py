"""Rule-based response generation for an emotion-aware robot."""

import re


RESPONSES = {
    "anger": (
        "That sounds frustrating. "
        "Want to tell me what happened?"
    ),

    "disgust": (
        "That sounds unpleasant. "
        "What bothered you about it?"
    ),

    "fear": (
        "That sounds worrying. "
        "Would you like to talk it through?"
    ),

    "joy": (
        "That's wonderful! "
        "What made it so exciting?"
    ),

    "neutral": (
        "I'm listening. Tell me more."
    ),

    "sadness": (
        "I'm sorry that's difficult. "
        "Would you like to talk about it?"
    ),

    "surprise": (
        "That sounds unexpected! "
        "What happened next?"
    )
}


def generate_response(text: str, emotion: str) -> str:
    """
    Generate a short conversational response based on
    the predicted emotion and simple text cues.
    """

    if emotion not in RESPONSES:
        raise ValueError(
            f"Unknown MELD emotion: {emotion}"
        )

    utterance = text.strip()

    if not utterance:
        return RESPONSES[emotion]

    lowered = utterance.lower()

    # Handle neutral questions
    if utterance.endswith("?") and emotion == "neutral":
        return (
            "That's a good question. "
            "Could you tell me a little more?"
        )

    # Handle expressions of gratitude
    if re.search(
        r"\b(thank you|thanks)\b",
        lowered
    ) and emotion in {"joy", "neutral"}:

        return "You're welcome!"

    # Handle farewells
    if re.search(
        r"\b(goodbye|bye|see you)\b",
        lowered
    ) and emotion == "neutral":

        return "See you later!"

    # Default response based on emotion
    return RESPONSES[emotion]