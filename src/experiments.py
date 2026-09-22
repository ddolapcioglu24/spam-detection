import os
import torch
import random
import numpy as np
import pandas as pd

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

def create_nested_training_subsets(
    train_df,
    training_sizes,
    random_seed=RANDOM_SEED,
):
    """
    Create reproducible, nested training subsets while preserving
    the original class distribution as closely as possible.
    """

    rng = np.random.default_rng(
        random_seed
    )

    # Shuffle ham and spam examples independently once.
    ham_df = train_df[
        train_df["label"] == "ham"
    ].copy()

    spam_df = train_df[
        train_df["label"] == "spam"
    ].copy()

    ham_order = rng.permutation(
        len(ham_df)
    )

    spam_order = rng.permutation(
        len(spam_df)
    )

    ham_df = ham_df.iloc[
        ham_order
    ].reset_index(drop=True)

    spam_df = spam_df.iloc[
        spam_order
    ].reset_index(drop=True)

    spam_ratio = (
        train_df["label"] == "spam"
    ).mean()

    subsets = {}

    for training_size in training_sizes:
        if training_size > len(train_df):
            raise ValueError(
                f"Training size {training_size} exceeds "
                f"the available training set size."
            )

        # Preserve the training-set class ratio.
        spam_count = int(
            round(
                training_size
                * spam_ratio
            )
        )

        ham_count = (
            training_size
            - spam_count
        )

        subset_df = pd.concat(
            [
                ham_df.iloc[:ham_count],
                spam_df.iloc[:spam_count],
            ],
            ignore_index=True,
        )

        # Shuffle the selected examples without changing membership.
        subset_df = subset_df.sample(
            frac=1,
            random_state=random_seed,
        ).reset_index(drop=True)

        subsets[
            training_size
        ] = subset_df

    return subsets

def get_model_checkpoint_path(
    base_dir,
    dataset_name,
    model_name,
):
    """
    Return the checkpoint path for a trained model.
    """

    dataset_names = {
        "UCI SMS": "uci_sms",
        "Turkish SMS": "turkish_sms",
        "YouTube Spam": "youtube_spam",
        "Enron Spam": "enron_spam",
    }

    model_names = {
        "BERT Base": "bert_base",
        "DistilBERT": "distilbert",
        "XLM-RoBERTa": "xlm_roberta",
        "TF-IDF + SVM": "tfidf_svm",
        "fastText": "fasttext",
    }

    dataset_dir = os.path.join(
        base_dir,
        dataset_names[dataset_name],
    )

    os.makedirs(
        dataset_dir,
        exist_ok=True,
    )

    model_key = model_names[model_name]

    if model_name == "TF-IDF + SVM":
        return os.path.join(
            dataset_dir,
            f"{model_key}.joblib",
        )

    if model_name == "fastText":
        return os.path.join(
            dataset_dir,
            f"{model_key}.bin",
        )

    # Transformer checkpoints are stored as directories.
    return os.path.join(
        dataset_dir,
        model_key,
    )

def run_same_dataset_experiment(
    dataset_splits,
    model_classes,
    results_path,
    model_dir,
):
    """
    Run same-dataset training and evaluation for all models.

    Parameters
    ----------
    dataset_splits : dict
        Mapping from dataset name to (train_df, test_df).
    model_classes : dict
        Mapping from model name to model class.
    results_path : str
        Path where experiment results are saved as CSV.
    model_dir : str
        Directory where trained model checkpoints are saved.

    Returns
    -------
    pandas.DataFrame
        Results for all completed runs.
    """

    os.makedirs(
        os.path.dirname(results_path),
        exist_ok=True,
    )

    os.makedirs(
        model_dir,
        exist_ok=True,
    )

    # Load previously completed results if available.
    if os.path.exists(results_path):
        results = (
            pd.read_csv(results_path)
            .to_dict("records")
        )
    else:
        results = []

    completed_runs = set()

    # A run is complete only if both its result and model exist.
    for result in results:
        checkpoint_path = get_model_checkpoint_path(
            model_dir,
            result["dataset"],
            result["model"],
        )

        if os.path.exists(checkpoint_path):
            completed_runs.add(
                (
                    result["dataset"],
                    result["model"],
                )
            )

    total_runs = (
        len(dataset_splits)
        * len(model_classes)
    )

    print(
        f"Completed runs: "
        f"{len(completed_runs)}/{total_runs}"
    )

    for dataset_name, (
        train_df,
        test_df,
    ) in dataset_splits.items():

        for model_name, model_class in model_classes.items():

            run_key = (
                dataset_name,
                model_name,
            )

            if run_key in completed_runs:
                print(
                    f"Skipping {dataset_name} | "
                    f"{model_name} "
                    f"(already completed)"
                )
                continue

            print(
                f"Running {model_name} "
                f"on {dataset_name}..."
            )

            set_random_seed()

            model = model_class()

            evaluation = evaluate_model(
                model,
                train_df,
                test_df,
            )

            checkpoint_path = get_model_checkpoint_path(
                model_dir,
                dataset_name,
                model_name,
            )

            model.save(
                checkpoint_path
            )

            # Remove an incomplete older result for the same run.
            results = [
                result
                for result in results
                if not (
                    result["dataset"] == dataset_name
                    and result["model"] == model_name
                )
            ]

            results.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    **evaluation,
                }
            )

            # Save immediately after every completed run.
            pd.DataFrame(
                results
            ).to_csv(
                results_path,
                index=False,
            )

            completed_runs.add(
                run_key
            )

            print(
                f"Saved {dataset_name} | "
                f"{model_name}"
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                torch.mps.empty_cache()

    return pd.DataFrame(
        results
    )

def run_transfer_experiment(
    dataset_splits,
    model_classes,
    results_path,
    model_dir,
):
    """
    Evaluate trained models across different datasets without retraining.

    Parameters
    ----------
    dataset_splits : dict
        Mapping from dataset name to (train_df, test_df).
    model_classes : dict
        Mapping from model name to model class.
    results_path : str
        Path where experiment results are saved as CSV.
    model_dir : str
        Directory containing Experiment 1 model checkpoints.

    Returns
    -------
    pandas.DataFrame
        Results for all completed transfer evaluations.
    """

    os.makedirs(
        os.path.dirname(results_path),
        exist_ok=True,
    )

    # Load previously completed results if available.
    if os.path.exists(results_path):
        results = (
            pd.read_csv(results_path)
            .to_dict("records")
        )
    else:
        results = []

    completed_runs = {
        (
            result["source_dataset"],
            result["target_dataset"],
            result["model"],
        )
        for result in results
    }

    total_runs = (
        len(dataset_splits)
        * (len(dataset_splits) - 1)
        * len(model_classes)
    )

    print(
        f"Completed transfer runs: "
        f"{len(completed_runs)}/{total_runs}"
    )

    for source_dataset in dataset_splits:
        for model_name, model_class in model_classes.items():

            # Find target datasets that still need evaluation.
            remaining_targets = [
                target_dataset
                for target_dataset in dataset_splits
                if (
                    target_dataset != source_dataset
                    and (
                        source_dataset,
                        target_dataset,
                        model_name,
                    )
                    not in completed_runs
                )
            ]

            if not remaining_targets:
                continue

            checkpoint_path = get_model_checkpoint_path(
                model_dir,
                source_dataset,
                model_name,
            )

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Missing checkpoint for "
                    f"{source_dataset} | {model_name}: "
                    f"{checkpoint_path}"
                )

            print(
                f"Loading {model_name} trained on "
                f"{source_dataset}..."
            )

            model = model_class()
            model.load(
                checkpoint_path
            )

            for target_dataset in remaining_targets:
                _, target_test_df = (
                    dataset_splits[
                        target_dataset
                    ]
                )

                print(
                    f"Evaluating {source_dataset} -> "
                    f"{target_dataset} | {model_name}"
                )

                evaluation = evaluate_trained_model(
                    model,
                    target_test_df,
                )

                results.append(
                    {
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "model": model_name,
                        **evaluation,
                    }
                )

                # Save immediately after every evaluation.
                pd.DataFrame(
                    results
                ).to_csv(
                    results_path,
                    index=False,
                )

                completed_runs.add(
                    (
                        source_dataset,
                        target_dataset,
                        model_name,
                    )
                )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                torch.mps.empty_cache()

    return pd.DataFrame(
        results
    )

def run_training_size_experiment(
    train_df,
    test_df,
    model_classes,
    training_sizes,
    results_path,
    dataset_name,
):
    """
    Evaluate model accuracy at different labelled training-set sizes.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Full training split used to create smaller subsets.
    test_df : pandas.DataFrame
        Fixed held-out test split used for every run.
    model_classes : dict
        Mapping from model name to model class.
    training_sizes : list
        Training-set sizes to evaluate.
    results_path : str
        Path where experiment results are saved as CSV.
    dataset_name : str
        Name of the dataset used for the experiment.

    Returns
    -------
    pandas.DataFrame
        Accuracy results for all completed runs.
    """

    os.makedirs(
        os.path.dirname(results_path),
        exist_ok=True,
    )

    training_subsets = create_nested_training_subsets(
        train_df,
        training_sizes,
    )

    # Load previously completed results if available.
    if os.path.exists(results_path):
        results = (
            pd.read_csv(results_path)
            .to_dict("records")
        )
    else:
        results = []

    completed_runs = {
        (
            result["model"],
            int(result["training_size"]),
        )
        for result in results
    }

    total_runs = (
        len(model_classes)
        * len(training_sizes)
    )

    print(
        f"Completed Experiment 3 runs: "
        f"{len(completed_runs)}/{total_runs}"
    )

    y_true = (
        test_df["label"] == "spam"
    ).astype(int).to_numpy()

    for model_name, model_class in model_classes.items():
        for training_size in training_sizes:

            run_key = (
                model_name,
                training_size,
            )

            if run_key in completed_runs:
                print(
                    f"Skipping {model_name} with "
                    f"{training_size} examples "
                    f"(already completed)"
                )
                continue

            print(
                f"Running {model_name} with "
                f"{training_size} training examples..."
            )

            train_subset = (
                training_subsets[
                    training_size
                ]
            )

            set_random_seed()

            model = model_class()

            model.fit(
                train_subset["text"].tolist(),
                train_subset["label"].tolist(),
            )

            y_score = model.predict_proba(
                test_df["text"].tolist()
            )

            y_pred = (
                y_score
                >= DECISION_THRESHOLD
            ).astype(int)

            accuracy = np.mean(
                y_true == y_pred
            )

            results.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "training_size": training_size,
                    "accuracy": accuracy,
                }
            )

            # Save immediately after every completed run.
            pd.DataFrame(
                results
            ).to_csv(
                results_path,
                index=False,
            )

            completed_runs.add(
                run_key
            )

            print(
                f"Saved {model_name} | "
                f"{training_size} examples | "
                f"accuracy={accuracy:.4f}"
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                torch.mps.empty_cache()

    return pd.DataFrame(
        results
    )