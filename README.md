# SMS Spam Detection

This project compares five different approaches to spam detection across multiple languages and text domains. The main goal is not only to compare predictive performance, but also to evaluate cross-dataset generalization, labelled-data requirements, realistic class imbalance, inference speed, model size, and classification errors.

The project is motivated by a practical deployment question: which model would be most suitable for large-scale spam filtering when the system must process large numbers of messages, generalize to previously unseen message types, and avoid blocking legitimate messages.

## Models

Five spam-detection approaches are evaluated:

1. **BERT Base** — `bert-base-uncased`
2. **DistilBERT** — `distilbert-base-uncased`
3. **XLM-RoBERTa** — `FacebookAI/xlm-roberta-base`
4. **TF-IDF character n-grams + Linear SVM**
5. **fastText**

An always-ham classifier is also included as a sanity baseline. It always predicts the legitimate-message class and demonstrates why accuracy alone can be misleading for spam detection.

All five main models follow a common interface:

- `fit(texts, labels)`
- `predict_proba(texts)`

The transformer models share their training and inference implementation through `TransformerSpamModel`.

## Datasets

Four datasets are used:

- **UCI SMS Spam Collection** — English SMS messages
- **Turkish SMS Collection** — Turkish SMS messages
- **UCI YouTube Spam Collection** — YouTube comments from a different text domain
- **Enron Spam** — English email messages from a different communication domain

### Dataset Statistics

| Dataset | Total | Ham | Spam | Spam Ratio |
|---|---:|---:|---:|---:|
| UCI SMS | 5,572 | 4,825 | 747 | 13.41% |
| Turkish SMS | 4,751 | 2,215 | 2,536 | 53.38% |
| YouTube Spam | 1,956 | 951 | 1,005 | 51.38% |
| Enron Spam | 33,716 | 16,545 | 17,171 | 50.93% |

No missing labels or message texts were found after loading the datasets.

## Duplicate and Overlap Checks

Before model training, messages are normalized for duplicate detection by:

- converting text to lowercase,
- removing punctuation,
- collapsing repeated whitespace.

### Internal Duplicates

| Dataset | Total Messages | Unique Normalized Messages | Duplicate Rows |
|---|---:|---:|---:|
| UCI SMS | 5,572 | 5,131 | 441 |
| Turkish SMS | 4,751 | 4,696 | 55 |
| YouTube Spam | 1,956 | 1,713 | 243 |
| Enron Spam | 33,716 | 30,101 | 3,615 |

For Enron, 3,222 duplicate rows already exist before normalization. Normalization introduces 393 additional duplicate rows.

The train/test splitting procedure keeps messages with identical normalized text in the same split so that duplicate messages cannot leak between training and test data.

### Cross-Dataset Overlap

| Dataset 1 | Dataset 2 | Shared Normalized Messages |
|---|---|---:|
| UCI SMS | Turkish SMS | 2 |
| UCI SMS | YouTube Spam | 1 |
| UCI SMS | Enron Spam | 0 |
| Turkish SMS | YouTube Spam | 1 |
| Turkish SMS | Enron Spam | 0 |
| YouTube Spam | Enron Spam | 0 |

Cross-dataset overlap is therefore very small, reducing the risk that transfer results are caused by identical messages appearing in multiple datasets.

## Evaluation Metrics

The main evaluation metrics are implemented manually in `src/metrics.py`:

- Precision
- Recall
- F1 score
- Precision-recall curve
- PR-AUC
- Precision at fixed recall

The default decision threshold is `0.50`, and precision at fixed recall is evaluated using a target recall of `0.90`.

Precision is especially important in this application because a false positive means that a legitimate message is incorrectly classified as spam.

Accuracy is used only where specifically required, such as the training-size experiment, and is not treated as the primary spam-detection metric.

## Experiments

### Experiment 1 — Same-Dataset Performance

Each of the five models is trained and evaluated on the same dataset using the fixed duplicate-safe train/test split.

The always-ham sanity baseline is also reported.

The main metrics are:

- Precision
- Recall
- F1
- PR-AUC
- Precision at 90% recall

All five approaches achieve strong in-domain performance.

The highest F1 score on each dataset is:

| Dataset | Highest-F1 Model | F1 |
|---|---|---:|
| UCI SMS | BERT Base | 0.9834 |
| Turkish SMS | TF-IDF + SVM | 0.9990 |
| YouTube Spam | DistilBERT | 0.9773 |
| Enron Spam | TF-IDF + SVM | 0.9932 |

One important result is that the TF-IDF + SVM model remains very close to the transformer models despite being dramatically smaller and faster.

All transformer models use a maximum sequence length of 256 tokens. This limitation is particularly relevant for Enron because many email messages are considerably longer and are therefore truncated during transformer processing.

### Experiment 2 — Cross-Dataset Transfer

Models trained in Experiment 1 are evaluated directly on every other dataset without retraining.

With four datasets and five models, this produces 60 cross-dataset evaluations.

Transfer performance varies substantially across dataset pairs.

These results should not be interpreted as a pure measurement of domain shift. Some transfers also involve:

- language shift,
- different message formats,
- vocabulary mismatch,
- tokenizer differences,
- pretrained-model mismatch.

For example, BERT Base and DistilBERT use English pretrained checkpoints even when evaluated on Turkish SMS. XLM-RoBERTa provides a multilingual comparison.

TF-IDF and fastText can also be strongly affected by changes in character patterns and vocabulary between datasets.

The experiment therefore shows that generalization to unseen data depends on both the model and the relationship between the source and target datasets.

### Experiment 3 — Training-Data Size

All five models are trained on increasingly large subsets of the UCI SMS training split:

- 50 examples
- 100 examples
- 500 examples
- 1,000 examples
- 2,000 examples

The same held-out UCI SMS test set is used for every training size.

An accuracy-versus-training-size plot is generated to compare sample efficiency.

The results generally improve as more labelled examples become available, but different models behave differently.

BERT Base and XLM-RoBERTa improve sharply between the smallest subsets and 500 examples, while the SVM already performs relatively strongly with small training sets. fastText improves more gradually.

These results should be treated as exploratory because each training size is evaluated using a single random seed. Multiple random seeds and error bars would be required for stronger conclusions about sample efficiency.

### Experiment 4 — Realistic Class Balance

The original test sets contain much higher spam proportions than real carrier traffic.

For this experiment:

- all legitimate test messages are retained,
- spam examples are downsampled,
- the resulting spam ratio is approximately 1%.

The same trained Experiment 1 models are then evaluated on the new imbalanced test sets.

This experiment demonstrates why precision becomes especially important when spam is rare. Even a small false-positive rate across a very large number of legitimate messages can produce many incorrect spam classifications.

### Experiment 5 — CPU Speed and Model Size

All five models are benchmarked on CPU using the same 1,000 UCI SMS test messages.

For each model:

1. the trained model is loaded,
2. several warm-up predictions are performed,
3. 1,000 messages are classified,
4. the benchmark is repeated five times,
5. the median throughput is reported.

Transformer inference uses the configured batch size of 16.

| Model | Messages / Second | Saved Model Size |
|---|---:|---:|
| BERT Base | 1.88 | 418 MB |
| DistilBERT | 3.78 | 256 MB |
| XLM-RoBERTa | 1.25 | 1,077 MB |
| TF-IDF + SVM | 2,199 | 2.6 MB |
| fastText | 8,269 | 768 MB |

DistilBERT is the most computationally efficient transformer in this benchmark.

However, the difference between the transformer models and the simpler models is several orders of magnitude. TF-IDF + SVM processes more than 2,000 messages per second while requiring only about 2.6 MB of storage.

This efficiency difference is important for large-scale deployment.

## Additional Analysis

### Pretrained Model Checkpoints

The same pretrained transformer checkpoints are used across all datasets:

| Model | Pretrained Checkpoint |
|---|---|
| BERT Base | `bert-base-uncased` |
| DistilBERT | `distilbert-base-uncased` |
| XLM-RoBERTa | `FacebookAI/xlm-roberta-base` |

BERT Base and DistilBERT are English pretrained models, while XLM-RoBERTa is multilingual.

This distinction is important when interpreting Turkish results.

### Tokenization Fragmentation Check

A sample of 500 messages from each dataset is tokenized using the BERT Base tokenizer.

Mean BERT tokens per whitespace-separated word:

| Dataset | Mean Tokens per Word |
|---|---:|
| UCI SMS | 1.54 |
| Turkish SMS | 3.05 |
| YouTube Spam | 4.95 |
| Enron Spam | 1.20 |

Turkish SMS requires substantially more BERT tokens per word than UCI SMS and Enron, supporting the concern that English BERT results on Turkish may partly reflect tokenizer and pretraining mismatch.

YouTube produces an even larger value because comments frequently contain URLs, usernames, unusual character sequences, and promotional strings.

The measurement should therefore be interpreted as evidence of tokenization mismatch rather than as a pure measurement of language difficulty.

### Precision-Recall Curves

Precision-recall curves are generated for all five models on the fixed UCI SMS test set.

All five models show strong precision-recall behavior, with TF-IDF + SVM remaining competitive with the transformer models across most recall values.

fastText shows a somewhat larger precision decrease as recall approaches 1.0.

### Confusion Matrices

For each dataset, the model with the highest Experiment 1 F1 score is selected for confusion-matrix analysis.

F1 is used as the selection criterion because it provides a single threshold-based score that balances precision and recall, allowing the models to be compared consistently across datasets.

Selected models:

- UCI SMS — BERT Base
- Turkish SMS — TF-IDF + SVM
- YouTube Spam — DistilBERT
- Enron Spam — TF-IDF + SVM

The confusion matrices confirm the strong same-dataset performance while also showing that the remaining types of error differ between datasets.

### Misclassification Analysis

Twenty misclassified messages from the selected highest-F1 models are sampled reproducibly and manually inspected.

The analysis shows several recurring patterns:

- legitimate messages can contain phone numbers, billing information, promotional phrases, or job-related content that resembles spam,
- spam messages can resemble normal business communication,
- short YouTube self-promotion can look similar to ordinary comments,
- some classifications are close to the `0.50` decision threshold,
- other errors occur with very high confidence,
- some dataset labels appear ambiguous or noisy.

This manual inspection demonstrates that aggregate metrics alone do not fully describe model behavior.

## Final Project Summary

The experiments show that all five approaches can achieve strong in-domain spam-classification performance, but their practical behavior differs substantially.

Transformer models achieve very high predictive performance, but TF-IDF + SVM remains competitive on several datasets while requiring far less storage and computation.

In the CPU benchmark, TF-IDF + SVM processes approximately 2,199 messages per second with a saved model size of only 2.6 MB, compared with approximately 1.88 messages per second and 418 MB for BERT Base.

The cross-dataset experiments also show that model performance can decrease substantially when models are applied to data from a different language or domain.

For this benchmark, TF-IDF + SVM provides a strong practical deployment trade-off when throughput and storage are major constraints because its predictive performance remains close to the transformer models while being dramatically faster and smaller.

However, the transfer experiments also show that deployment decisions should consider how much the target language, vocabulary, and message domain may change over time.

## Project Structure

```text
spam-detection/
├── data/
│   ├── raw/                 # Raw datasets, excluded from Git
│   └── processed/
│
├── cache/                   # Trained models, excluded from Git
│
├── notebooks/
│   └── spam_detection.ipynb
│
├── outputs/                 # Experiment tables and plots
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── experiments.py
│   │
│   └── models/
│       ├── transformer_base.py
│       ├── bert_base.py
│       ├── distilbert.py
│       ├── xlm_roberta.py
│       ├── svm.py
│       └── fasttext_model.py
│
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ddolapcioglu24/spam-detection.git
cd spam-detection
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The main pinned dependencies are:

- `accelerate==1.10.1`
- `datasets==4.5.0`
- `fasttext==0.9.3`
- `matplotlib==3.9.4`
- `numpy==1.26.4`
- `pandas==2.3.3`
- `scikit-learn==1.6.1`
- `torch==2.8.0`
- `transformers==4.57.6`

## Dataset Setup

Raw datasets are intentionally excluded from the Git repository.

Place the dataset files under the following structure:

```text
data/
└── raw/
    ├── uci_sms/
    │   └── SMSSpamCollection
    │
    ├── turkish_sms/
    │   └── TurkishSMSCollection.csv
    │
    ├── youtube_spam/
    │   ├── Youtube01-Psy.csv
    │   ├── Youtube02-KatyPerry.csv
    │   ├── Youtube03-LMFAO.csv
    │   ├── Youtube04-Eminem.csv
    │   └── Youtube05-Shakira.csv
    │
    └── enron/
        └── enron_spam.csv
```

These locations correspond to the dataset paths defined in `src/config.py`.

The provided `raw_data.zip` archive can also be extracted directly into the repository root so that it recreates the `data/raw/` structure.

## Running the Project

The notebook contains the complete experimental workflow.

Open:

```text
notebooks/spam_detection.ipynb
```

and execute the cells in order.

Google Colab is recommended for transformer fine-tuning because GPU execution substantially reduces training time.

The notebook is designed to:

- reuse experiment results that already exist,
- reuse saved trained models where appropriate,
- avoid repeating completed expensive runs,
- save experiment results and plots for later analysis.

Some notebook cells use Google Drive for persistent checkpoints and outputs when running in Colab. If using a different Drive location, update those paths accordingly.

## Outputs

The `outputs/` directory contains the main experimental results.

It should include the five experiment result tables:

```text
outputs/
├── experiment_1_results.csv
├── experiment_2_results.csv
├── experiment_3_results.csv
├── experiment_4_results.csv
├── experiment_5_results.csv
```

Additional generated outputs include:

```text
experiment_3_training_size_accuracy.png
uci_sms_pr_curves.png
best_model_confusion_matrices.png
misclassified_messages_sample.csv
```

The required plots are therefore:

- accuracy versus training size,
- precision-recall curves for all five models on UCI SMS.

The confusion-matrix and misclassification-analysis outputs provide additional qualitative evaluation.

## Reproducibility

The main experiment configuration is defined in `src/config.py`.

Key settings include:

- random seed: `42`
- test size: `0.20`
- decision threshold: `0.50`
- target recall: `0.90`
- transformer batch size: `16`
- transformer maximum sequence length: `256`
- transformer fine-tuning epochs: `3`

Train/test splits are duplicate-safe and reproducible.

Experiment 3 uses nested training subsets generated with the same random seed.

## Limitations

Several limitations should be considered when interpreting the results.

First, BERT Base and DistilBERT use English pretrained checkpoints on every dataset, including Turkish SMS. Their Turkish results therefore combine spam-classification performance with a language and tokenizer mismatch.

Second, cross-dataset transfer can involve several changes simultaneously, including language, domain, message length, vocabulary, and spam style. A decrease in transfer performance should therefore not automatically be attributed to one specific cause.

Third, Experiment 3 uses only one random seed for each training size. The results are useful as an exploratory comparison, but repeated runs with multiple seeds and error bars would provide stronger evidence about sample efficiency.

Finally, transformer models use a shared maximum sequence length of 256 tokens. Long Enron emails may therefore be truncated, meaning that transformer results on Enron reflect only the first part of many messages.