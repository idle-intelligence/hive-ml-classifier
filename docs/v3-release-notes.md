# v3 Model Release Notes

## What changed

### 2-class → 4-class classification

The model now classifies into **4 categories** instead of 2:

| Label ID | Label | Description |
|----------|-------|-------------|
| 0 | `other` | General queries, chitchat, coding help, greetings, etc. |
| 1 | `swarm` | Anything about the swarm system: peers, nodes, workers, users, observers, tok/s, questions, messages, etc. |
| 2 | `time` | Time, date, timers, countdowns, scheduling, uptime |
| 3 | `weather` | Weather, temperature, forecasts, conditions |

The output JSON now returns scores for all 4 classes:

```json
{
  "scores": { "other": 0.04, "swarm": 0.89, "time": 0.03, "weather": 0.03 },
  "label": "swarm",
  "score": 0.89
}
```

### What the hive team needs to change

1. **Update label handling.** The `id2label` map in `config.json` now has 4 entries instead of 2. If your Rust code reads `num_labels` from config (as recommended), it should Just Work — the classification head is `[4, 256]` instead of `[2, 256]`. But you'll need to update the routing logic to handle the new labels.

2. **Update the LABELS array in Rust** (if hardcoded). Change from:
   ```rust
   const LABELS: &[&str] = &["general", "swarm"];
   ```
   to:
   ```rust
   const LABELS: &[&str] = &["other", "swarm", "time", "weather"];
   ```
   Or better: read labels from `config.json` at runtime.

3. **"general" is now "other".** The label name changed. Update any code that checks for `label == "general"`.

4. **Routing logic.** You now get 4 categories. Suggested routing:
   - `swarm` → route to swarm (same as before)
   - `weather` → route to weather handler
   - `time` → route to time handler
   - `other` → route to general handler

   For observers: `swarm` → "join the swarm to know the swarm", everything else routes normally.

5. **Tokenizer fix.** The tokenizer now has `lowercase: true` in the normalizer. The v1/v2 tokenizers shipped with `lowercase: false`, which meant the Rust tokenizer wasn't lowercasing input before tokenization. If your Rust code was manually lowercasing input as a workaround, you can remove that — the tokenizer handles it now.

6. **Model size is unchanged.** Still bert-mini (11M params), same 43MB safetensors, same 256 hidden_size. No WASM rebuild needed for the model architecture — only the classification head dimensions changed (2→4 outputs).

### Question-format bias fix

The v1/v2 model had a bias where question-format inputs ("How many...", "What is...", "Tell me about...") tended to classify as general/other regardless of content. This happened because the training data had swarm examples mostly as commands and general examples mostly as questions.

v3 fixes this with:
- 225 question-format swarm examples (43% of swarm class)
- Balanced question/statement format across all classes
- Specific examples like "How many peers are in the swarm?", "What's your tok/s?", "Tell me about the swarm!" now correctly classify as swarm with 80-91% confidence

### Performance

| Metric | v1 (2-class) | v2 (4-class) | v3 (4-class) |
|--------|-------------|-------------|-------------|
| Accuracy | 92.2% | 87.6% | **94.4%** |
| F1 (weighted) | 91.6% | 87.5% | **94.3%** |
| Swarm recall | 94.2% | 90.0% | **98.0%** |
| Edge cases | 84.0% | 89.7% | **94.8%** |

Per-class F1: other 0.96, swarm 0.95, time 0.84, weather 0.95.

### Delivery files

```
delivery-v3/
├── model.safetensors   (43 MB, bert-mini 4-layer 256-hidden)
├── config.json         (915 B, num_labels=4, id2label mapping)
└── tokenizer.json      (695 KB, BertNormalizer with lowercase=true)
```

### v3.1 hotfix — safetensors naming + config fix

Two issues reported by the hive team have been fixed in the delivery files:

1. **LayerNorm tensor naming.** The `prajjwal1/bert-mini` base model uses the old BERT convention (`gamma`/`beta`). The v3 safetensors now uses modern naming (`weight`/`bias`) which Candle expects. The training pipeline has also been patched to auto-rename on save, so future deliveries won't have this issue.

2. **`num_labels` added to config.json.** The field was missing, which caused the Rust code to default to `num_labels=2`. Now explicitly set to `4`. You can remove any workarounds that derive `num_labels` from `id2label` length — though keeping that fallback is fine.

### Known weaknesses

- "what questions are being asked" → classifies as other (ambiguous without "swarm" in text)
- "do I need an umbrella" → classifies as other (indirect weather reference)
- Single-word ambiguous inputs like "hot" have lower confidence
- Time class has slightly lower recall (80%) — some time examples get confused with other

### Training data

1508 examples (527 swarm, 616 other, 179 time, 186 weather). Trained for 10 epochs on bert-mini with lr=3e-5, batch_size=32, max_length=64.
