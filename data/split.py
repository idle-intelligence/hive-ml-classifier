"""Create 80/20 train/val split with seed 42. Run from data/ directory."""
import csv
import random
import os

SEED = 42
SPLIT_RATIO = 0.8
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Read dataset
rows = []
with open(os.path.join(DATA_DIR, "dataset.csv"), newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Shuffle with fixed seed
random.seed(SEED)
random.shuffle(rows)

# Split
split_idx = int(len(rows) * SPLIT_RATIO)
train_rows = rows[:split_idx]
val_rows = rows[split_idx:]

# Write splits
for name, data in [("train.csv", train_rows), ("val.csv", val_rows)]:
    with open(os.path.join(DATA_DIR, name), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(data)

# Print statistics
print(f"Total examples: {len(rows)}")
print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")
from collections import Counter
for split_name, data in [("Train", train_rows), ("Val", val_rows)]:
    c = Counter(r["label"] for r in data)
    parts = [f"{label}={count} ({count/len(data)*100:.1f}%)" for label, count in sorted(c.items())]
    print(f"  {split_name}: {', '.join(parts)}")
