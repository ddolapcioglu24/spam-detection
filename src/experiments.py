import os
import time
import torch
import random
import numpy as np
import pandas as pd

from src.metrics import precision_recall_curve
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

def create_realistic_test_set(
    test_df,
    target_spam_ratio=0.01,
    random_seed=RANDOM_SEED,
):
    """
    Create a test set with a target spam ratio by downsampling spam.

    All ham messages are retained, while spam messages are randomly
    downsampled to approximate the requested class distribution.

    Parameters
    ----------
    test_df : pandas.DataFrame
        Test set containing ``text`` and ``label`` columns.
    target_spam_ratio : float, optional
        Desired spam ratio in the resulting test set.
    random_seed : int, optional
        Random seed used when sampling spam messages.

    Returns
    -------
    pandas.DataFrame
        Test set with all ham messages and downsampled spam messages.
    """
    ham_df = test_df[
        test_df["label"] == "ham"
    ]

    spam_df = test_df[
        test_df["label"] == "spam"
    ]

    # Compute how many spam messages give approximately the target ratio.
    target_spam_count = round(
        (
            target_spam_ratio
            * len(ham_df)
        )
        / (
            1.0
            - target_spam_ratio
        )
    )

    # Downsample spam while keeping every ham message.
    target_spam_count = min(
        target_spam_count,
        len(spam_df),
    )

    sampled_spam_df = spam_df.sample(
        n=target_spam_count,
        random_state=random_seed,
    )

    realistic_test_df = pd.concat(
        [
            ham_df,
            sampled_spam_df,
        ],
        ignore_index=True,
    )

    # Shuffle the final test set reproducibly.
    return realistic_test_df.sample(
        frac=1.0,
        random_state=random_seed,
    ).reset_index(drop=True)

def run_realistic_balance_experiment(
    realistic_test_sets,
    model_classes,
    results_path,
    model_dir,
):
    """
    Evaluate trained models on test sets with realistic spam prevalence.

    Parameters
    ----------
    realistic_test_sets : dict
        Mapping from dataset name to realistic test DataFrame.
    model_classes : dict
        Mapping from model name to model class.
    results_path : str
        Path where experiment results are saved as CSV.
    model_dir : str
        Directory containing Experiment 1 model checkpoints.

    Returns
    -------
    pandas.DataFrame
        Results for all completed realistic-balance evaluations.
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
            result["dataset"],
            result["model"],
        )
        for result in results
    }

    total_runs = (
        len(realistic_test_sets)
        * len(model_classes)
    )

    print(
        f"Completed Experiment 4 runs: "
        f"{len(completed_runs)}/{total_runs}"
    )

    for dataset_name, test_df in realistic_test_sets.items():
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

            checkpoint_path = get_model_checkpoint_path(
                model_dir,
                dataset_name,
                model_name,
            )

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Missing checkpoint for "
                    f"{dataset_name} | {model_name}: "
                    f"{checkpoint_path}"
                )

            print(
                f"Evaluating {model_name} "
                f"on realistic {dataset_name}..."
            )

            model = model_class()
            model.load(
                checkpoint_path
            )

            evaluation = evaluate_trained_model(
                model,
                test_df,
            )

            results.append(
                {
                    "dataset": dataset_name,
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

def get_checkpoint_size_bytes(checkpoint_path):
    """
    Return the total size of a model checkpoint in bytes.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Single-file checkpoints are used by SVM and fastText.
    if os.path.isfile(checkpoint_path):
        return os.path.getsize(
            checkpoint_path
        )

    total_size = 0

    # Transformer checkpoints are stored as directories.
    for root, _, files in os.walk(checkpoint_path):
        for file_name in files:
            file_path = os.path.join(
                root,
                file_name,
            )

            total_size += os.path.getsize(
                file_path
            )

    return total_size


def move_model_to_cpu(model):
    """
    Move a loaded model to CPU when the model uses a PyTorch device.
    """

    cpu_device = torch.device("cpu")

    # Transformer wrappers store the selected device here.
    if hasattr(model, "device"):
        model.device = cpu_device

    # Move the underlying PyTorch model itself to CPU.
    if (
        hasattr(model, "model")
        and hasattr(model.model, "to")
    ):
        model.model.to(cpu_device)

    return model


def run_speed_size_experiment(
    benchmark_df,
    model_classes,
    results_path,
    model_dir,
    dataset_name,
    num_messages=1000,
    warmup_size=20,
    num_runs=5,
):
    """
    Measure CPU inference throughput and checkpoint size for each model.

    Parameters
    ----------
    benchmark_df : pandas.DataFrame
        Dataset containing messages used for the throughput benchmark.
    model_classes : dict
        Mapping from model name to model class.
    results_path : str
        Path where Experiment 5 results are saved as CSV.
    model_dir : str
        Directory containing Experiment 1 model checkpoints.
    dataset_name : str
        Dataset whose trained checkpoints are benchmarked.
    num_messages : int, optional
        Number of messages used in each timed run.
    warmup_size : int, optional
        Number of messages used before timing begins.
    num_runs : int, optional
        Number of timed runs used to compute median throughput.

    Returns
    -------
    pandas.DataFrame
        CPU throughput and model-size results.
    """

    if num_messages < 1000:
        raise ValueError(
            "Experiment 5 requires at least 1000 benchmark messages."
        )

    if len(benchmark_df) < num_messages:
        raise ValueError(
            f"Benchmark dataset contains only {len(benchmark_df)} "
            f"messages, but {num_messages} were requested."
        )

    if warmup_size < 1:
        raise ValueError(
            "warmup_size must be at least 1."
        )

    if num_runs < 1:
        raise ValueError(
            "num_runs must be at least 1."
        )

    os.makedirs(
        os.path.dirname(results_path),
        exist_ok=True,
    )

    benchmark_texts = (
        benchmark_df["text"]
        .iloc[:num_messages]
        .tolist()
    )

    warmup_texts = benchmark_texts[
        :min(warmup_size, num_messages)
    ]

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
            result["dataset"],
            result["model"],
            int(result["num_messages"]),
            int(result["num_runs"]),
        )
        for result in results
    }

    total_runs = len(model_classes)

    print(
        f"Completed Experiment 5 runs: "
        f"{len(completed_runs)}/{total_runs}"
    )

    for model_name, model_class in model_classes.items():

        run_key = (
            dataset_name,
            model_name,
            num_messages,
            num_runs,
        )

        if run_key in completed_runs:
            print(
                f"Skipping {model_name} "
                f"(already completed)"
            )
            continue

        checkpoint_path = get_model_checkpoint_path(
            model_dir,
            dataset_name,
            model_name,
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Missing checkpoint for "
                f"{dataset_name} | {model_name}: "
                f"{checkpoint_path}"
            )

        print(
            f"Benchmarking {model_name} on CPU..."
        )

        # Load the trained Experiment 1 checkpoint.
        model = model_class()
        model.load(
            checkpoint_path
        )

        # Force transformer inference onto CPU.
        model = move_model_to_cpu(
            model
        )

        # Warm up the model before timing.
        model.predict_proba(
            warmup_texts
        )

        throughputs = []

        # Measure throughput several times using the same message batch.
        for _ in range(num_runs):
            start_time = time.perf_counter()

            model.predict_proba(
                benchmark_texts
            )

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            throughputs.append(
                num_messages / elapsed_time
            )

        median_throughput = float(
            np.median(throughputs)
        )

        checkpoint_size_bytes = (
            get_checkpoint_size_bytes(
                checkpoint_path
            )
        )

        model_size_mb = (
            checkpoint_size_bytes
            / (1024 ** 2)
        )

        results.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "num_messages": num_messages,
                "num_runs": num_runs,
                "messages_per_second": median_throughput,
                "model_size_mb": model_size_mb,
            }
        )

        # Save immediately after every completed benchmark.
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
            f"{median_throughput:.2f} messages/sec | "
            f"{model_size_mb:.2f} MB"
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(
        results
    )

def get_model_predictions(
    model,
    test_df,
    decision_threshold=DECISION_THRESHOLD,
):
    """
    Generate labels, spam scores, and hard predictions for a test set.

    Parameters
    ----------
    model
        Trained spam detection model.
    test_df : pandas.DataFrame
        Test set containing ``text`` and ``label`` columns.
    decision_threshold : float, optional
        Threshold used to convert spam scores into class predictions.

    Returns
    -------
    tuple
        Ground-truth labels, spam scores, and predicted labels.
    """

    # Convert test labels to binary values: ham=0, spam=1.
    y_true = (
        test_df["label"] == "spam"
    ).astype(int).to_numpy()

    # Generate spam probabilities or comparable prediction scores.
    y_score = model.predict_proba(
        test_df["text"].tolist()
    )

    # Convert scores to hard class predictions.
    y_pred = (
        y_score >= decision_threshold
    ).astype(int)

    return y_true, y_score, y_pred

def get_pr_curves_for_dataset(
    test_df,
    model_classes,
    model_dir,
    dataset_name,
):
    """
    Generate precision-recall curve data for all models on one dataset.
    """

    curve_data = {}

    for model_name, model_class in model_classes.items():
        # Load the trained checkpoint for the selected dataset.
        checkpoint_path = get_model_checkpoint_path(
            model_dir,
            dataset_name,
            model_name,
        )

        model = model_class()
        model.load(checkpoint_path)

        # Generate prediction scores for the fixed test set.
        y_true, y_score, _ = get_model_predictions(
            model,
            test_df,
        )

        # Compute precision and recall values across thresholds.
        precisions, recalls = precision_recall_curve(
            y_true,
            y_score,
        )

        curve_data[model_name] = {
            "precision": precisions,
            "recall": recalls,
        }

        # Release the loaded model before moving to the next one.
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return curve_data

def get_confusion_matrix_for_model(
    test_df,
    model_class,
    model_dir,
    dataset_name,
    model_name,
):
    """
    Generate a confusion matrix for one trained model on one dataset.
    """

    # Load the trained checkpoint for the selected dataset.
    checkpoint_path = get_model_checkpoint_path(
        model_dir,
        dataset_name,
        model_name,
    )

    model = model_class()
    model.load(checkpoint_path)

    # Generate true and predicted labels for the fixed test set.
    y_true, _, y_pred = get_model_predictions(
        model,
        test_df,
    )

    # Count the four possible prediction outcomes.
    true_negative = np.sum(
        (y_true == 0) & (y_pred == 0)
    )

    false_positive = np.sum(
        (y_true == 0) & (y_pred == 1)
    )

    false_negative = np.sum(
        (y_true == 1) & (y_pred == 0)
    )

    true_positive = np.sum(
        (y_true == 1) & (y_pred == 1)
    )

    confusion_matrix = np.array(
        [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ]
    )

    # Release the loaded model after prediction.
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return confusion_matrix