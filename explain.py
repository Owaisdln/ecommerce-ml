def build_explanation(product_id, context_bonus, emotion_bonus, content_score):
    explanation_parts = []

    # Collaborative part (always present)
    explanation_parts.append(
        "Recommended based on collaborative filtering patterns from similar users."
    )

    # Context explanation
    if context_bonus > 0:
        explanation_parts.append(
            f"Adjusted for selected context settings (+{round(context_bonus,3)})."
        )

    # Emotion explanation
    if emotion_bonus > 0.1:
        explanation_parts.append(
            f"Strongly influenced by your emotional input (+{round(emotion_bonus,3)})."
        )
    elif emotion_bonus > 0:
        explanation_parts.append(
            f"Slightly aligned with your emotional input (+{round(emotion_bonus,3)})."
        )

    # Hybrid explanation
    if content_score > 0.2:
        explanation_parts.append(
            "Content similarity played a significant role in ranking."
        )
    elif content_score > 0:
        explanation_parts.append(
            "Partially matched product content similarity."
        )

    return " ".join(explanation_parts)