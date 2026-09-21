import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.config import (
    SVM_NGRAM_RANGE,
    SVM_MIN_DF,
    SVM_MAX_FEATURES,
    SVM_C,
    SVM_CALIBRATION_CV,
)


class TfidfSVMModel:
    """
    Spam classifier based on character-level TF-IDF features
    and a Linear Support Vector Machine.
    """

    def __init__(
        self,
        ngram_range=SVM_NGRAM_RANGE,
        min_df=SVM_MIN_DF,
        max_features=SVM_MAX_FEATURES,
        C=SVM_C,
        calibration_cv=SVM_CALIBRATION_CV,
    ):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
        )

        # Create the Linear SVM classifier.
        base_model = LinearSVC(
        C=C,
)

        # Calibrate the SVM decision scores to obtain probabilities.
        self.model = CalibratedClassifierCV(
            base_model,
            method="sigmoid",
            cv=calibration_cv,
        )

    def fit(self, texts, labels):
        """
        Train the TF-IDF vectorizer and Linear SVM classifier.
        """
        # Learn the character n-gram vocabulary and transform the messages.
        features = self.vectorizer.fit_transform(texts)

        # Convert string labels to numerical labels:
        # ham = 0, spam = 1.
        numeric_labels = np.array([
            1 if label == "spam" else 0
            for label in labels
        ])

        # Train the calibrated Linear SVM using the TF-IDF feature vectors.
        self.model.fit(
            features,
            numeric_labels,
        )

    def predict_proba(self, texts):
        """
        Predict the probability that each input message is spam.
        """
        # Transform new messages using the vocabulary learned during training.
        features = self.vectorizer.transform(texts)

        # Generate class probabilities for each message.
        probabilities = self.model.predict_proba(features)

        # Return only the probability of the spam class.
        spam_probabilities = probabilities[:, 1]

        return spam_probabilities