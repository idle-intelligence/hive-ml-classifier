#!/usr/bin/env python3
"""Fine-tune prajjwal1/bert-mini for 4-class classification (weather, swarm, time, other)."""

import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

LABEL_MAP = {"other": 0, "swarm": 1, "time": 2, "weather": 3}
MODEL_NAME = "prajjwal1/bert-mini"
MAX_LENGTH = 64
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_dataset_from_csv(path: str) -> Dataset:
    df = pd.read_csv(path)
    df["label"] = df["label"].map(LABEL_MAP)
    return Dataset.from_pandas(df[["text", "label"]])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_MAP),
        id2label={v: k for k, v in LABEL_MAP.items()},
        label2id=LABEL_MAP,
    )

    train_ds = load_dataset_from_csv(os.path.join(DATA_DIR, "train.csv"))
    val_ds = load_dataset_from_csv(os.path.join(DATA_DIR, "val.csv"))

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Fix LayerNorm naming: prajjwal1/bert-mini uses old gamma/beta convention.
    # Candle and modern HuggingFace expect weight/bias.
    safetensors_path = os.path.join(OUTPUT_DIR, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file, save_file

        tensors = load_file(safetensors_path)
        needs_rename = any(".gamma" in k or ".beta" in k for k in tensors)
        if needs_rename:
            renamed = {
                k.replace(".LayerNorm.gamma", ".LayerNorm.weight")
                .replace(".LayerNorm.beta", ".LayerNorm.bias"): v
                for k, v in tensors.items()
            }
            save_file(renamed, safetensors_path)
            print("Fixed LayerNorm naming: gamma/beta → weight/bias")

    print(f"\nModel saved to {OUTPUT_DIR}")
    metrics = trainer.evaluate()
    print(f"Final eval metrics: {metrics}")


if __name__ == "__main__":
    main()
