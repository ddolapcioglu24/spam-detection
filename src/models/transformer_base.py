import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.utils.data import Dataset, DataLoader


class SpamDataset(Dataset):
    """
    PyTorch dataset for transformer-based spam classification.

    It converts raw text messages into tokenized model inputs and
    optionally includes numerical labels during training.
    """

    def __init__(self, texts, tokenizer, max_length, labels=None):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # Retrieve one raw message.
        text = self.texts[index]

        # Convert the message into transformer input tokens.
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove the extra batch dimension added by the tokenizer.
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        # Labels are included during training but omitted during prediction.
        if self.labels is not None:
            item["label"] = torch.tensor(
                self.labels[index],
                dtype=torch.long,
            )

        return item
    
class TransformerSpamModel:
    """
    Common base class for transformer-based spam classifiers.

    It handles model loading, device selection, training, and prediction.
    Specific transformer models only need to provide their model name
    and experiment configuration.
    """

    def __init__(
        self,
        model_name,
        max_length,
        batch_size,
        epochs,
        learning_rate,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Select the best available device.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load the tokenizer corresponding to the pretrained model.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        # Load the pretrained transformer with a binary classification head.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
        )

        # Move the model to the selected computation device.
        self.model.to(self.device)

    def fit(self, texts, labels):
        """
        Fine-tune the transformer model for binary spam classification.
        """

        # Convert string labels to numerical labels:
        # ham = 0, spam = 1.
        numeric_labels = [
            1 if label == "spam" else 0
            for label in labels
        ]

        # Create the training dataset.
        train_dataset = SpamDataset(
            texts=texts,
            labels=numeric_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        # Create shuffled mini-batches for training.
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # AdamW is used to update the transformer parameters.
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # Enable training behavior such as dropout.
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for batch in train_loader:
                # Move the current batch to the same device as the model.
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["label"].to(self.device)

                # Remove gradients left over from the previous batch.
                optimizer.zero_grad()

                # Forward pass. Providing labels makes the model
                # calculate the classification loss automatically.
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels,
                )

                loss = outputs.loss

                # Compute gradients with backpropagation.
                loss.backward()

                # Update the model parameters using the gradients.
                optimizer.step()

                # Accumulate the loss only for training progress reporting.
                total_loss += loss.item()

            average_loss = total_loss / len(train_loader)

            print(
                f"Epoch {epoch + 1}/{self.epochs} "
                f"- Average Loss: {average_loss:.4f}"
            )

    def predict_proba(self, texts):
        """
        Predict the probability that each input message is spam.
        """

        # Create a dataset without labels for inference.
        prediction_dataset = SpamDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        # Preserve the original message order during prediction.
        prediction_loader = DataLoader(
            prediction_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # Switch the model to evaluation mode.
        self.model.eval()

        # Store spam probabilities from all batches.
        all_probabilities = []

        # Gradients are not required during inference.
        with torch.no_grad():
            for batch in prediction_loader:
                # Move the current batch to the model's device.
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward pass without labels.
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Convert raw classification logits into probabilities.
                probabilities = torch.softmax(
                    outputs.logits,
                    dim=1,
                )

                # Class 1 represents spam.
                spam_probabilities = probabilities[:, 1]

                # Move predictions back to CPU before converting to NumPy.
                spam_probabilities = spam_probabilities.cpu().numpy()

                all_probabilities.append(spam_probabilities)

        # Combine predictions from all batches into one array.
        return np.concatenate(all_probabilities)