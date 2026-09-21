import torch
import random
import numpy as np

from src.data import normalize_text
from src.config import (
    RANDOM_SEED,
    TEST_SIZE,
    DECISION_THRESHOLD,
    TARGET_RECALL,
)

from src.metrics import (
    precision_score,
    recall_score,
    f1_score,
    pr_auc_score,
    precision_at_recall,
)


def create_train_test_split(
    df,
    test_size=TEST_SIZE,
    random_seed=RANDOM_SEED,
):
    """
    Create a reproducible, duplicate-safe train/test split.

    Messages with the same normalized text are kept in the same split
    while preserving the row-level label distribution as closely as possible.
    """

    # Create a copy so the original dataset remains unchanged.
    split_df = df.copy()

    # Use normalized text to identify duplicate message groups.
    split_df["normalized_text"] = (
        split_df["text"].map(normalize_text)
    )

    # Store the label and number of rows represented by each message group.
    groups = (
        split_df
        .groupby("normalized_text", as_index=False)
        .agg(
            label=("label", "first"),
            group_size=("label", "size"),
        )
    )

    rng = np.random.default_rng(random_seed)
    test_group_names = []

    # Select test groups separately for each label.
    for label in ["ham", "spam"]:
        label_groups = groups[
            groups["label"] == label
        ].copy()

        order = rng.permutation(len(label_groups))
        label_groups = label_groups.iloc[order]

        total_rows = label_groups["group_size"].sum()
        target_rows = total_rows * test_size
        selected_rows = 0

        # Add groups while they improve the distance to the target size.
        for _, group in label_groups.iterrows():
            current_difference = abs(
                target_rows - selected_rows
            )
            new_difference = abs(
                target_rows
                - (selected_rows + group["group_size"])
            )

            if new_difference <= current_difference:
                test_group_names.append(
                    group["normalized_text"]
                )
                selected_rows += group["group_size"]

    # Assign complete message groups to train or test.
    is_test = split_df["normalized_text"].isin(
        test_group_names
    )

    train_df = split_df[~is_test].copy()
    test_df = split_df[is_test].copy()

    # Remove the temporary grouping column.
    train_df = train_df.drop(
        columns="normalized_text"
    ).reset_index(drop=True)

    test_df = test_df.drop(
        columns="normalized_text"
    ).reset_index(drop=True)

    return train_df, test_df

def always_ham_predictions(size):
    """
    Create predictions that classify every message as ham.
    """

    return np.zeros(size, dtype=int)

def set_random_seed(seed=RANDOM_SEED):
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluate_trained_model(
    model,
    test_df,
    decision_threshold=DECISION_THRESHOLD,
    target_recall=TARGET_RECALL,
):
    """
    Evaluate an already trained model on a test dataset.
    """

    # Convert test labels to binary values: ham=0, spam=1.
    y_true = (
        test_df["label"] == "spam"
    ).astype(int).to_numpy()

    # Get spam probabilities or comparable scores.
    y_score = model.predict_proba(
        test_df["text"].tolist()
    )

    # Convert scores to hard predictions at the common threshold.
    y_pred = (
        y_score >= decision_threshold
    ).astype(int)

    # Compute the required evaluation metrics.
    results = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "pr_auc": pr_auc_score(y_true, y_score),
        "precision_at_recall": precision_at_recall(
            y_true,
            y_score,
            target_recall,
        ),
    }

    return results

def evaluate_model(
    model,
    train_df,
    test_df,
    decision_threshold=DECISION_THRESHOLD,
    target_recall=TARGET_RECALL,
):
    """
    Train and evaluate one model on a given train/test split.
    """

    # Train the model on the training messages.
    model.fit(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
    )

    # Evaluate the trained model on the test set.
    return evaluate_trained_model(
        model,
        test_df,
        decision_threshold=decision_threshold,
        target_recall=target_recall,
    )