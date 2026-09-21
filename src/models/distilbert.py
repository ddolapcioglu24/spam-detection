from src.config import (
    DISTILBERT_MODEL_NAME,
    DISTILBERT_MAX_LENGTH,
    DISTILBERT_BATCH_SIZE,
    DISTILBERT_EPOCHS,
    DISTILBERT_LEARNING_RATE,
)

from src.models.transformer_base import TransformerSpamModel


class DistilBERTModel(TransformerSpamModel):
    """
    Spam classifier based on the pretrained DistilBERT Base Uncased model.
    """

    def __init__(
        self,
        model_name=DISTILBERT_MODEL_NAME,
        max_length=DISTILBERT_MAX_LENGTH,
        batch_size=DISTILBERT_BATCH_SIZE,
        epochs=DISTILBERT_EPOCHS,
        learning_rate=DISTILBERT_LEARNING_RATE,
    ):
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
        )