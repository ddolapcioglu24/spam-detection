from src.config import (
    XLM_ROBERTA_MODEL_NAME,
    XLM_ROBERTA_MAX_LENGTH,
    XLM_ROBERTA_BATCH_SIZE,
    XLM_ROBERTA_EPOCHS,
    XLM_ROBERTA_LEARNING_RATE,
)

from src.models.transformer_base import TransformerSpamModel


class XLMRoBERTaModel(TransformerSpamModel):
    """
    Spam classifier based on the pretrained XLM-RoBERTa Base model.
    """

    def __init__(
        self,
        model_name=XLM_ROBERTA_MODEL_NAME,
        max_length=XLM_ROBERTA_MAX_LENGTH,
        batch_size=XLM_ROBERTA_BATCH_SIZE,
        epochs=XLM_ROBERTA_EPOCHS,
        learning_rate=XLM_ROBERTA_LEARNING_RATE,
    ):
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
        )