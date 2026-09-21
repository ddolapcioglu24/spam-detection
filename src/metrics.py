def precision_score(y_true, y_pred):
    """
    Compute precision for the spam class.
    """

    # Count correctly and incorrectly predicted spam messages.
    true_positive = 0
    false_positive = 0

    for true_label, predicted_label in zip(y_true, y_pred):
        if predicted_label == 1:
            if true_label == 1:
                true_positive += 1
            else:
                false_positive += 1

    # Avoid division by zero if no message is predicted as spam.
    if true_positive + false_positive == 0:
        return 0.0

    return true_positive / (true_positive + false_positive)

def recall_score(y_true, y_pred):
    """
    Compute recall for the spam class.
    """

    # Count detected and missed spam messages.
    true_positive = 0
    false_negative = 0

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == 1:
            if predicted_label == 1:
                true_positive += 1
            else:
                false_negative += 1

    # Avoid division by zero if there is no spam in the true labels.
    if true_positive + false_negative == 0:
        return 0.0

    return true_positive / (true_positive + false_negative)

def f1_score(y_true, y_pred):
    """
    Compute F1 score for the spam class.
    """

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Avoid division by zero if both precision and recall are zero.
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def precision_recall_curve(y_true, y_score):
    """
    Compute precision and recall values at different score thresholds.
    """

    # Sort messages from highest to lowest spam score.
    sorted_pairs = sorted(
        zip(y_score, y_true),
        reverse=True,
    )

    total_positive = sum(y_true)
    true_positive = 0
    false_positive = 0

    precisions = [1.0]
    recalls = [0.0]

    i = 0

    while i < len(sorted_pairs):
        current_score = sorted_pairs[i][0]

        # Process all messages with the same score together.
        while (
            i < len(sorted_pairs)
            and sorted_pairs[i][0] == current_score
        ):
            true_label = sorted_pairs[i][1]

            if true_label == 1:
                true_positive += 1
            else:
                false_positive += 1

            i += 1

        precision = (
            true_positive
            / (true_positive + false_positive)
        )

        if total_positive == 0:
            recall = 0.0
        else:
            recall = true_positive / total_positive

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def pr_auc_score(y_true, y_score):
    """
    Compute the area under the precision-recall curve.
    """

    precisions, recalls = precision_recall_curve(
        y_true,
        y_score,
    )

    area = 0.0

    # Sum the trapezoid areas between consecutive curve points.
    for i in range(1, len(recalls)):
        recall_difference = recalls[i] - recalls[i - 1]
        average_precision = (
            precisions[i] + precisions[i - 1]
        ) / 2

        area += recall_difference * average_precision

    return area

def precision_at_recall(y_true, y_score, target_recall):
    """
    Compute the highest precision at or above a target recall.
    """

    precisions, recalls = precision_recall_curve(
        y_true,
        y_score,
    )

    valid_precisions = []

    # Keep precision values that satisfy the recall target.
    for precision, recall in zip(precisions, recalls):
        if recall >= target_recall:
            valid_precisions.append(precision)

    if not valid_precisions:
        return 0.0

    return max(valid_precisions)