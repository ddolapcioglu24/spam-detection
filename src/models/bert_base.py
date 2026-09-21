from src.config import (
    BERT_BASE_MODEL_NAME,
    BERT_MAX_LENGTH,
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LEARNING_RATE,
)
from src.models.transformer_base import TransformerSpamModel
   

class BERTBaseModel(TransformerSpamModel):
    def __init__(
        self,
        model_name=BERT_BASE_MODEL_NAME,
        max_length=BERT_MAX_LENGTH,
        batch_size=BERT_BATCH_SIZE,
        epochs=BERT_EPOCHS,
        learning_rate=BERT_LEARNING_RATE,
    ):
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
        )