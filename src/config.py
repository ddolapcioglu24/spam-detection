from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset paths
UCI_SMS_PATH = RAW_DATA_DIR / "uci_sms" / "SMSSpamCollection"
TURKISH_SMS_PATH = RAW_DATA_DIR / "turkish_sms" / "TurkishSMSCollection.csv"
YOUTUBE_SPAM_DIR = RAW_DATA_DIR / "youtube_spam"
ENRON_SPAM_PATH = RAW_DATA_DIR / "enron" / "enron_spam.csv"

# Other project directories
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# -------------------------
# BERT Configuration
# -------------------------

BERT_BASE_MODEL_NAME = "bert-base-uncased"

BERT_MAX_LENGTH = 256
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 3
BERT_LEARNING_RATE = 2e-5

# -------------------------
# DistilBERT Configuration
# -------------------------

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"

DISTILBERT_MAX_LENGTH = 256
DISTILBERT_BATCH_SIZE = 16
DISTILBERT_EPOCHS = 3
DISTILBERT_LEARNING_RATE = 2e-5

# -------------------------
# XLM-RoBERTa Configuration
# -------------------------

XLM_ROBERTA_MODEL_NAME = "FacebookAI/xlm-roberta-base"

XLM_ROBERTA_MAX_LENGTH = 256
XLM_ROBERTA_BATCH_SIZE = 16
XLM_ROBERTA_EPOCHS = 3
XLM_ROBERTA_LEARNING_RATE = 2e-5

# -------------------------
# TF-IDF + Linear SVM Configuration
# -------------------------

SVM_NGRAM_RANGE = (2, 5)
SVM_MIN_DF = 2
SVM_MAX_FEATURES = 100000
SVM_C = 1.0
SVM_CALIBRATION_CV = 5

# -------------------------
# fastText Configuration
# -------------------------

FASTTEXT_EPOCHS = 25
FASTTEXT_LR = 0.1
FASTTEXT_WORD_NGRAMS = 2
FASTTEXT_DIM = 100
FASTTEXT_MIN_N = 3
FASTTEXT_MAX_N = 6

# Experiment settings
RANDOM_SEED = 42
TEST_SIZE = 0.20
DECISION_THRESHOLD = 0.50
TARGET_RECALL = 0.90