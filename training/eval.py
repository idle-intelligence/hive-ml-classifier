#!/usr/bin/env python3
"""Evaluate the fine-tuned 4-class classifier on val.csv and hardcoded edge cases."""

import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_MAP = {"other": 0, "swarm": 1, "time": 2, "weather": 3}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "output")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MAX_LENGTH = 64

EDGE_CASES = [
    # Clear swarm
    ("restart the cluster", "swarm"),
    ("how many peers are connected", "swarm"),
    ("check node health", "swarm"),
    ("ping the mesh", "swarm"),
    ("list active workers", "swarm"),
    ("show swarm dashboard", "swarm"),
    ("tell me about the swarm", "swarm"),
    ("tell me about the swarm!", "swarm"),
    ("how many users are online", "swarm"),
    ("show me the observers", "swarm"),
    ("what's the tok/s", "swarm"),
    ("what questions are being asked", "swarm"),
    ("how many observers are watching", "swarm"),
    ("what's up with the swarm", "swarm"),
    ("who are the active users", "swarm"),
    ("tokens per second", "swarm"),
    # Clear weather
    ("what's the weather like", "weather"),
    ("is it going to rain", "weather"),
    ("temperature outside", "weather"),
    ("do I need an umbrella", "weather"),
    ("will it snow tomorrow", "weather"),
    ("how cold is it", "weather"),
    ("weather forecast", "weather"),
    ("is it sunny", "weather"),
    # Clear time
    ("what time is it", "time"),
    ("what's the date today", "time"),
    ("how many days until Christmas", "time"),
    ("what day of the week is it", "time"),
    ("current time in UTC", "time"),
    ("how long until Friday", "time"),
    ("set a timer for 5 minutes", "time"),
    ("when is sunset", "time"),
    # Clear other
    ("write me a poem", "other"),
    ("tell me a joke", "other"),
    ("help me debug my code", "other"),
    ("what is machine learning", "other"),
    ("translate this to French", "other"),
    ("how do I reverse a linked list", "other"),
    ("what's the capital of France", "other"),
    ("explain quantum computing", "other"),
    # Ambiguous / boundary
    ("what's up", "other"),
    ("what's up?", "other"),
    ("status", "swarm"),
    ("help", "other"),
    ("how's it going", "other"),
    ("what's happening", "other"),
    ("when", "time"),
    ("now", "time"),
    ("hot", "weather"),
    ("cold", "weather"),
    ("what is a mesh network", "other"),
    ("explain peer-to-peer networking", "other"),
    ("is it warm enough for a picnic", "weather"),
    ("how long has the session been running", "time"),
    ("uptime", "time"),
    ("how many users", "swarm"),
    ("observers", "swarm"),
    ("tok/s", "swarm"),
]


def classify(text: str, model, tokenizer) -> dict:
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    scores = {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(len(LABEL_MAP))}
    pred_idx = probs.argmax().item()
    return {"scores": scores, "label": ID2LABEL[pred_idx], "score": probs[pred_idx].item()}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # --- Validation set evaluation ---
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    true_labels = [LABEL_MAP[l] for l in val_df["label"]]
    preds = []
    for text in val_df["text"]:
        result = classify(text, model, tokenizer)
        preds.append(LABEL_MAP[result["label"]])

    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average="weighted"
    )
    cm = confusion_matrix(true_labels, preds)

    print("=" * 60)
    print("VALIDATION SET RESULTS")
    print("=" * 60)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"  Labels: {list(LABEL_MAP.keys())}")
    print(f"  {cm}")
    print(f"\n{classification_report(true_labels, preds, target_names=list(LABEL_MAP.keys()))}")

    # --- Edge cases ---
    print("=" * 60)
    print("EDGE CASE RESULTS")
    print("=" * 60)
    edge_results = []
    edge_correct = 0
    for text, expected in EDGE_CASES:
        result = classify(text, model, tokenizer)
        correct = result["label"] == expected
        edge_correct += int(correct)
        marker = "OK" if correct else "FAIL"
        print(f"  [{marker}] \"{text}\"")
        print(f"         expected={expected}  got={result['label']}  scores={result['scores']}")
        edge_results.append({
            "text": text,
            "expected": expected,
            "predicted": result["label"],
            "scores": result["scores"],
            "correct": correct,
        })

    edge_acc = edge_correct / len(EDGE_CASES)
    print(f"\nEdge case accuracy: {edge_correct}/{len(EDGE_CASES)} ({edge_acc:.1%})")

    # --- Save results ---
    results = {
        "validation": {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "n_samples": len(true_labels),
        },
        "edge_cases": {
            "accuracy": edge_acc,
            "n_correct": edge_correct,
            "n_total": len(EDGE_CASES),
            "details": edge_results,
        },
    }

    results_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
