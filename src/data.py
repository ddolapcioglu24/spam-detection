import re
import ast
import string

from pathlib import Path
from src.config import (
    UCI_SMS_PATH,
    TURKISH_SMS_PATH,
    YOUTUBE_SPAM_DIR,
    ENRON_SPAM_PATH,
)
import pandas as pd

def normalize_text(text):
    """
    Normalize message text for duplicate detection.

    The normalization follows the assignment:
    - convert to lowercase
    - remove punctuation
    - collapse extra whitespace

    Parameters
    ----------
    text : str
        Original message text.

    Returns
    -------
    str
        Normalized message text.
    """
    text = str(text).lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text

def load_uci_sms(path=UCI_SMS_PATH):
    """
    Load the UCI SMS Spam Collection.

    Parameters
    ----------
    path : str or Path
        Path to the SMSSpamCollection file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:
        - label: "ham" or "spam"
        - text: message text
    """
    path = Path(path)

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
    )

    return df

def load_turkish_sms(path=TURKISH_SMS_PATH):
    """
    Load the Turkish SMS Collection.

    Parameters
    ----------
    path : str or Path
        Path to the TurkishSMSCollection.csv file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:
        - label: "ham" or "spam"
        - text: message text
    """
    path = Path(path)

    df = pd.read_csv(
        path,
        sep=";"
    )

    df = df[["GroupText", "Message"]].copy()

    df = df.rename(
        columns={
            "GroupText": "label",
            "Message": "text"
        }
    )

    df["label"] = (
        df["label"]
        .str.lower()
        .replace({"normal": "ham"})
    )

    return df

def load_youtube_spam(path=YOUTUBE_SPAM_DIR):
    """
    Load and combine the five UCI YouTube Spam Collection files.

    Parameters
    ----------
    path : str or Path
        Directory containing the YouTube spam CSV files.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:
        - label: "ham" or "spam"
        - text: comment text
    """
    path = Path(path)

    csv_files = sorted(path.glob("Youtube*.csv"))

    dataframes = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        df = df[["CLASS", "CONTENT"]].copy()

        df = df.rename(
            columns={
                "CLASS": "label",
                "CONTENT": "text"
            }
        )

        df["label"] = df["label"].map({
            0: "ham",
            1: "spam"
        })

        dataframes.append(df)

    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    return combined_df

def load_enron_spam(path=ENRON_SPAM_PATH):
    """
    Load the Enron Spam dataset.

    Parameters
    ----------
    path : str or Path
        Path to the enron_spam.csv file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:
        - label: "ham" or "spam"
        - text: email text
    """
    path = Path(path)

    df = pd.read_csv(path)

    df = df[["label", "email"]].copy()

    df = df.rename(
        columns={
            "email": "text"
        }
    )

    df["label"] = df["label"].map({
        0: "ham",
        1: "spam"
    })

    df["text"] = df["text"].map(
        lambda text: ast.literal_eval(text)[0]
    )

    return df

def get_duplicate_stats(df):
    """
    Calculate duplicate statistics within one dataset
    after normalizing the message text.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing a "text" column.

    Returns
    -------
    dict
        Duplicate statistics for the dataset.
    """
    normalized = df["text"].map(normalize_text)

    return {
        "total_messages": len(df),
        "unique_messages": normalized.nunique(),
        "duplicate_rows": int(normalized.duplicated().sum()),
    }


def get_dataset_overlap(df1, df2):
    """
    Find unique normalized messages appearing in both datasets.

    Parameters
    ----------
    df1 : pandas.DataFrame
        First dataset containing a "text" column.

    df2 : pandas.DataFrame
        Second dataset containing a "text" column.

    Returns
    -------
    set
        Unique normalized messages shared by both datasets.
    """
    texts1 = set(df1["text"].map(normalize_text))
    texts2 = set(df2["text"].map(normalize_text))

    return texts1.intersection(texts2)