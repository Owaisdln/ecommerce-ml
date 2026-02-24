def apply_context_adjustment(predictions, context_selection):

    adjusted = []

    for product_id, score in predictions:

        context_bonus = 0

        # Budget
        if context_selection.get("budget") == "Low":
            context_bonus += 0.04
        elif context_selection.get("budget") == "Medium":
            context_bonus += 0.05
        elif context_selection.get("budget") == "High":
            context_bonus += 0.03
        elif context_selection.get("budget") == "Premium":
            context_bonus += 0.02

        # Occasion
        if context_selection.get("occasion") == "Casual":
            context_bonus += 0.02
        elif context_selection.get("occasion") == "Party":
            context_bonus += 0.05
        elif context_selection.get("occasion") == "Work":
            context_bonus += 0.04
        elif context_selection.get("occasion") == "Travel":
            context_bonus += 0.03
        elif context_selection.get("occasion") == "Study":
            context_bonus += 0.03
        elif context_selection.get("occasion") == "Gift":
            context_bonus += 0.04
        elif context_selection.get("occasion") == "Festival":
            context_bonus += 0.05

        # Time
        if context_selection.get("time") == "Morning":
            context_bonus += 0.01
        elif context_selection.get("time") == "Afternoon":
            context_bonus += 0.02
        elif context_selection.get("time") == "Evening":
            context_bonus += 0.03
        elif context_selection.get("time") == "Night":
            context_bonus += 0.02
        elif context_selection.get("time") == "Weekend":
            context_bonus += 0.04

        final_score = score + context_bonus
        adjusted.append((product_id, final_score, context_bonus))

    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted