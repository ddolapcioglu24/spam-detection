import os
import tempfile

import fasttext
import numpy as np

from src.config import (
    FASTTEXT_EPOCHS,
    FASTTEXT_LR,
    FASTTEXT_WORD_NGRAMS,
    FASTTEXT_DIM,
    FASTTEXT_MIN_N,
    FASTTEXT_MAX_N,
)


class FastTextModel:
    """
    Spam classifier based on the fastText supervised learning model.
    """

    def __init__(
        self,
        epochs=FASTTEXT_EPOCHS,
        learning_rate=FASTTEXT_LR,
        word_ngrams=FASTTEXT_WORD_NGRAMS,
        dim=FASTTEXT_DIM,
        min_n=FASTTEXT_MIN_N,
        max_n=FASTTEXT_MAX_N,
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.word_ngrams = word_ngrams
        self.dim = dim
        self.min_n = min_n
        self.max_n = max_n
        self.model = None

    def fit(self, texts, labels):
        """
        Train the fastText model for binary spam classification.
        """
        # Create a temporary training file in fastText format.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            temp_path = temp_file.name

            for text, label in zip(texts, labels):
                # Keep each training example on a single line.

                clean_text = str(text).replace("\n", " ").strip()

                temp_file.write(
                    f"__label__{label} {clean_text}\n"
                )

        try:
            # Train the supervised fastText classifier.
            self.model = fasttext.train_supervised(
                input=temp_path,
                epoch=self.epochs,
                lr=self.learning_rate,
                wordNgrams=self.word_ngrams,
                dim=self.dim,
                minn=self.min_n,
                maxn=self.max_n,
            )

        finally:
            # Remove the temporary training file.
            os.remove(temp_path)

    def predict_proba(self, texts):
        """
        Predict the probability that each input message is spam.
        """
        if self.model is None:
            raise RuntimeError(
                "The model must be trained before prediction."
            )

        spam_probabilities = []

        for text in texts:
            # Keep each input message on a single line.
            clean_text = str(text).replace("\n", " ").strip()

            labels, probabilities = self.model.predict(
                [clean_text],
                k=2,
            )

            # Find the probability assigned to the spam class.
            label_probabilities = dict(
                zip(labels[0], probabilities[0])
            )

            spam_probability = label_probabilities.get(
                "__label__spam",
                0.0,
            )

            spam_probabilities.append(
                spam_probability
            )

        return np.array(spam_probabilities)
    
    def save(self, path):
        """
        Save the trained fastText model.
        """
        if self.model is None:
            raise RuntimeError(
                "The model must be trained before saving."
            )

        self.model.save_model(path)

    def load(self, path):
        """
        Load a previously trained fastText model.
        """
        self.model = fasttext.load_model(path)