# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML classifier for the Hive swarm system. Routes user input into four categories (swarm, weather, time, other) using a fine-tuned BERT model running client-side in the browser via Rust/WASM.

This repo is the ML side only: dataset generation (`data/`), training (`training/`), and
model deliveries (`delivery*/`, config + tokenizer; weights ship separately). The Rust/WASM
inference pipeline and web worker live in the consuming Hive app, not here.

## Architecture (consuming app)

```
Main Thread → postMessage → Web Worker (WASM) → classify → postMessage → Routing
```

The worker loads the model once on init, then classifies each message in ~2-10ms. No network round-trip, no GPU. The classifier consumes three files (`model.safetensors`, `config.json`, `tokenizer.json`) and returns `{ scores, label, score }`.

Two user types affect routing:
- **Observers** — swarm-classified input is blocked ("join the swarm to know the swarm"), general input routes normally
- **Swarm members** — input is tagged by the swarm (with local classifier as fallback on timeout)

## Build Commands

```bash
# Generate and split the dataset
python data/generate.py
python data/split.py

# Train the model
python training/train.py
python training/eval.py

# Inspect safetensors keys (debugging weight name mismatches)
python -c "from safetensors import safe_open; f = safe_open('model.safetensors', framework='pt'); [print(k, f.get_tensor(k).shape) for k in f.keys()]"
```

## Key Design Decisions

- **bert-mini (prajjwal1/bert-mini, 4 layers, 256 hidden, ~11M params, ~45MB)** is the current model. Avoid distilbert-base (~250MB) — too large for WASM.
- **Full fine-tuning**, not LoRA — models are small enough and we want a single self-contained safetensors file.
- **NUM_LABELS must be read from config.json**, not hardcoded in Rust. This allows model swaps (4-class → 12-class) without WASM rebuilds.
- **Return full probability distribution** from `classify()`, not just the top label. Routing layer owns threshold/fallback logic.
- **Tokenizer max_length=64** during training. Inference uses truncation only (batch size always 1, no padding needed).
- Pin Candle dependency versions — it's pre-1.0 and APIs change between releases.

## Roadmap Context

Phase 1: Binary classifier (swarm/general) — done. Phase 2 (current): 4-class classifier (swarm, weather, time, other). Phase 3: Full 12-class taxonomy per swarm data layer.

## Two-Team Development Model

- **Hive Team** builds the inference pipeline (Rust/WASM, web worker, routing integration) using an off-the-shelf model
- **ML Team** builds the model (dataset generation, training scripts, evaluation) and delivers the three model files

The interface contract between teams is the three model files in, JSON classification result out. Weight-only updates are file swaps with no code changes.
