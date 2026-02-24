from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def apply_emotion_adjustment(predictions, user_text, product_dict):
    """
    predictions: list of tuples (product_id, score, context_bonus)
    user_text: mood input
    product_dict: mapping product_id -> product_name
    """

    sentiment = analyzer.polarity_scores(user_text)
    compound = sentiment['compound']  # value between -1 and +1

    adjusted = []

    for product_id, score, context_bonus in predictions:

        product_name = product_dict.get(product_id, "").lower()

        # Base scaling factor (controls strength of emotion impact)
        intensity = abs(compound)

        emotion_bonus = 0

        # positive mood
        if compound > 0.3:
            if any(keyword in product_name for keyword in ["echo", "music", "fire", "entertainment"]):
                emotion_bonus = 0.3 * intensity
            else:
                emotion_bonus = 0.1 * intensity

        # negative mood
        elif compound < -0.3:
            if any(keyword in product_name for keyword in ["kindle", "book", "paperwhite"]):
                emotion_bonus = 0.3 * intensity
            else:
                emotion_bonus = 0.1 * intensity

        # neutral
        else:
            emotion_bonus = 0.05

        final_score = score + emotion_bonus

        adjusted.append((product_id, final_score, context_bonus, emotion_bonus))

    adjusted.sort(key=lambda x: x[1], reverse=True)

    return adjusted