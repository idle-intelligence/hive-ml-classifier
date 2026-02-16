# Client-Side Classification with Candle

Fine-tune a small BERT classifier in Python, export to safetensors, run inference in Rust/WASM via Candle in a web worker.

---

## Why This Approach

- **CPU-only, no WebGPU dependency.** WASM runs everywhere, no browser feature flags or GPU requirements.
- **Tiny and fast.** A TinyBERT or DistilBERT classifier is ~15-60MB. Inference is ~2-10ms per classification.
- **Rust-native stack.** Candle + HuggingFace `tokenizers` crate — no JS ML runtime, no ONNX intermediate format. Safetensors in, prediction out.
- **Lean bundle.** Candle WASM compiles to ~2-5MB. Compare to ONNX Runtime WASM at ~8-10MB or WebLLM at hundreds of MB.
- **Fits the swarm.** The rest of the project is Rust. This keeps the inference code in the same language, debuggable with the same tools.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  Main Thread                                │
│                                             │
│  user types message                         │
│       │                                     │
│       ▼                                     │
│  postMessage({ text: "restart cluster" })   │
│       │                                     │
└───────┼─────────────────────────────────────┘
        │
        │  (Web Worker boundary)
        │
┌───────▼─────────────────────────────────────┐
│  Worker (WASM)                              │
│                                             │
│  ┌──────────────┐    ┌──────────────────┐   │
│  │  tokenizers   │───▶│  candle (BERT)    │  │
│  │  (Rust/WASM)  │    │  + classification │  │
│  │              │    │    head           │  │
│  └──────────────┘    └───────┬──────────┘   │
│                              │              │
│                              ▼              │
│               { tag: "swarm", score: 0.97 } │
│                              │              │
└──────────────────────────────┼──────────────┘
                               │
                               ▼
                    postMessage({ tag, score })
                               │
                               ▼
                     route to swarm / general
```

The worker loads the model once on init, then classifies each message in ~2-10ms. No round-trip, no network, no GPU.

---

## Step 1: Prepare Training Data

You need labeled examples. A few hundred is enough for binary classification with a pretrained encoder.

### Format

```csv
text,label
"restart the cluster",swarm
"what's the weather like",general
"show me connected peers",swarm
"write a poem about cats",general
"how many nodes are online",swarm
"ping the hub",swarm
"translate this to french",general
```

### How to Bootstrap

**Option A: Generate with Claude.** Prompt Claude to produce 200-300 examples per category. Be specific about edge cases:

```
Generate 200 example user inputs for a swarm/mesh networking system.
Half should be "swarm" (about the system itself: peers, nodes, connections, routing, hub, workers, mesh status).
Half should be "general" (anything else: questions, creative tasks, coding help, chitchat).
Include edge cases: ambiguous inputs, short inputs, inputs that mention tech words but aren't about the swarm.
Format as CSV: text,label
```

**Option B: Collect real inputs.** Log actual user inputs from the swarm app, label them manually. Best quality, but requires existing usage.

**Option C: Hybrid.** Generate synthetic data, then refine with real examples as they come in. Start with synthetic, retrain later.

Target: **300-500 examples minimum.** Split 80/20 for train/validation.

---

## Step 2: Fine-Tune the Model

We fine-tune a small pretrained BERT variant with a classification head on top. LoRA is unnecessary here — the models are small enough for full fine-tuning, and we want a single self-contained model file at the end.

### Model Selection

| Model                  | Params | Size (safetensors) | Notes                              |
|------------------------|--------|--------------------|------------------------------------|
| `google/bert_uncased_L-2_H-128_A-2` | ~4M   | ~17MB             | Tiny BERT. Probably enough.        |
| `prajjwal1/bert-tiny`  | ~4M    | ~17MB              | Same architecture, different training. |
| `prajjwal1/bert-mini`  | ~11M   | ~45MB              | More capacity if tiny isn't enough. |
| `distilbert-base-uncased` | ~66M | ~250MB            | Overkill for binary classification, but bulletproof. |

Start with **bert-tiny**. If accuracy is bad, step up to bert-mini. Don't go to DistilBERT unless you really need to — the size jump is significant for a WASM bundle.

### Training Script

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import pandas as pd

# --- Config ---
MODEL_NAME = "prajjwal1/bert-tiny"
OUTPUT_DIR = "./swarm-classifier"
NUM_LABELS = 2
LABEL_MAP = {"general": 0, "swarm": 1}

# --- Load data ---
df = pd.read_csv("training_data.csv")
df["label"] = df["label"].map(LABEL_MAP)

# Split 80/20
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df).map(tokenize, batched=True),
    "validation": Dataset.from_pandas(val_df).map(tokenize, batched=True),
})

# --- Model ---
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# --- Train ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,          # small dataset, more epochs
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()

# --- Save ---
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")
print(f"Files: {list(os.listdir(OUTPUT_DIR))}")
```

This produces a directory with:
- `model.safetensors` — the weights
- `config.json` — model architecture config
- `tokenizer.json` — the tokenizer (used by HuggingFace `tokenizers` crate)
- `tokenizer_config.json`, `vocab.txt`, etc.

### Quick Validation

```python
from transformers import pipeline

classifier = pipeline("text-classification", model=OUTPUT_DIR)

tests = [
    "restart the cluster",
    "how many peers are connected",
    "write me a haiku",
    "check node health",
    "what's 2 + 2",
    "route this to the hub",
]

for text in tests:
    result = classifier(text)
    print(f"{text:40s} → {result[0]['label']:10s} ({result[0]['score']:.3f})")
```

If accuracy on the validation set is >95%, you're good to go. If not, add more training examples or step up to bert-mini.

---

## Step 3: Candle Inference in Rust

This is the Rust side. We load the safetensors model and tokenizer, run a forward pass, and return the classification.

### Project Setup

```toml
# Cargo.toml
[package]
name = "swarm-classifier"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]    # for WASM

[dependencies]
candle-core = { version = "0.8", features = ["wasm"] }
candle-nn = "0.8"
candle-transformers = "0.8"
tokenizers = { version = "0.20", default-features = false, features = ["unstable_wasm"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
wasm-bindgen = "0.2"
```

Note: check the latest candle version — the API is still evolving. The `wasm` feature on candle-core enables the WASM backend.

### Model Loading and Inference

```rust
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

const LABELS: &[&str] = &["general", "swarm"];

#[wasm_bindgen]
pub struct Classifier {
    model: BertModel,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    tokenizer: Tokenizer,
    device: Device,
}

#[wasm_bindgen]
impl Classifier {
    /// Load model from bytes (fetched by JS side).
    /// model_bytes: the model.safetensors content
    /// config_bytes: the config.json content
    /// tokenizer_bytes: the tokenizer.json content
    #[wasm_bindgen(constructor)]
    pub fn new(
        model_bytes: &[u8],
        config_bytes: &[u8],
        tokenizer_bytes: &[u8],
    ) -> Result<Classifier, JsError> {
        let device = Device::Cpu;

        // Parse config
        let config: Config = serde_json::from_slice(config_bytes)
            .map_err(|e| JsError::new(&format!("config parse error: {e}")))?;

        // Load weights from safetensors
        let vb = VarBuilder::from_buffered_safetensors(
            vec![model_bytes.to_vec()],
            DType::F32,
            &device,
        ).map_err(|e| JsError::new(&format!("weight load error: {e}")))?;

        // Load BERT encoder
        let model = BertModel::load(vb.pp("bert"), &config)
            .map_err(|e| JsError::new(&format!("model load error: {e}")))?;

        // Load classification head weights
        // HuggingFace saves these as "classifier.weight" and "classifier.bias"
        let classifier_weight = vb.get(&[config.num_labels, config.hidden_size], "classifier.weight")
            .map_err(|e| JsError::new(&format!("classifier weight error: {e}")))?;
        let classifier_bias = vb.get(config.num_labels, "classifier.bias")
            .map_err(|e| JsError::new(&format!("classifier bias error: {e}")))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| JsError::new(&format!("tokenizer error: {e}")))?;

        Ok(Classifier { model, classifier_weight, classifier_bias, tokenizer, device })
    }

    /// Classify a single input string.
    /// Returns JSON: { "scores": { "general": 0.03, "swarm": 0.97 }, "label": "swarm", "score": 0.97 }
    pub fn classify(&self, text: &str) -> Result<String, JsError> {
        // Tokenize
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| JsError::new(&format!("tokenize error: {e}")))?;

        let token_ids = encoding.get_ids();
        let type_ids = encoding.get_type_ids();

        let token_ids = Tensor::new(token_ids, &self.device)?
            .unsqueeze(0)?;  // batch dim
        let type_ids = Tensor::new(type_ids, &self.device)?
            .unsqueeze(0)?;

        // Forward pass through BERT
        let output = self.model.forward(&token_ids, &type_ids, None)?;

        // Take [CLS] token embedding (first token)
        let cls_output = output.get(0)?.get(0)?;  // [hidden_size]

        // Classification head: linear layer
        let logits = cls_output
            .matmul(&self.classifier_weight.t()?)?
            .broadcast_add(&self.classifier_bias)?;

        // Softmax
        let probs = candle_nn::ops::softmax(&logits, 0)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // Find best label
        let (best_idx, best_score) = probs_vec.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        // Build full scores map
        let scores: serde_json::Map<String, serde_json::Value> = LABELS.iter()
            .zip(probs_vec.iter())
            .map(|(label, score)| (label.to_string(), serde_json::json!(score)))
            .collect();

        let result = serde_json::json!({
            "scores": scores,
            "label": LABELS[best_idx],
            "score": best_score,
        });

        Ok(result.to_string())
    }
}
```

**Important caveats on this code:**
- The exact Candle BERT API may differ slightly depending on the version. The `BertModel` in `candle-transformers` expects a specific config format — HuggingFace's `config.json` should map directly, but you may need to adjust field names.
- The classification head loading (`classifier.weight`, `classifier.bias`) depends on how HuggingFace's `AutoModelForSequenceClassification` names the layers in safetensors. Inspect with `safetensors.safe_open()` in Python to verify the key names.
- The `tokenizers` crate with `unstable_wasm` feature is what enables WASM compilation. This feature flag may change.

### Build to WASM

```bash
# Install wasm-pack if you haven't
cargo install wasm-pack

# Build
wasm-pack build --target web --release

# Output lands in ./pkg/
# ├── swarm_classifier_bg.wasm    (~2-5MB)
# ├── swarm_classifier.js          (JS glue)
# └── swarm_classifier.d.ts        (TypeScript types)
```

---

## Step 4: Web Worker Integration

### Worker Script

```javascript
// classifier-worker.js
import init, { Classifier } from './pkg/swarm_classifier.js';

let classifier = null;

self.onmessage = async (e) => {
    const { type, payload } = e.data;

    if (type === 'init') {
        // Initialize WASM and load model
        await init();

        // Fetch model files (cache these — they don't change)
        const [modelBytes, configBytes, tokenizerBytes] = await Promise.all([
            fetch('/models/model.safetensors').then(r => r.arrayBuffer()).then(b => new Uint8Array(b)),
            fetch('/models/config.json').then(r => r.arrayBuffer()).then(b => new Uint8Array(b)),
            fetch('/models/tokenizer.json').then(r => r.arrayBuffer()).then(b => new Uint8Array(b)),
        ]);

        classifier = new Classifier(modelBytes, configBytes, tokenizerBytes);
        self.postMessage({ type: 'ready' });
    }

    if (type === 'classify') {
        if (!classifier) {
            self.postMessage({ type: 'error', error: 'not initialized' });
            return;
        }
        const result = classifier.classify(payload.text);
        self.postMessage({
            type: 'result',
            id: payload.id,
            result: JSON.parse(result),
        });
    }
};
```

### Main Thread API

```javascript
// classifier.js
export function createClassifier() {
    const worker = new Worker(
        new URL('./classifier-worker.js', import.meta.url),
        { type: 'module' }
    );

    let readyResolve;
    const ready = new Promise(resolve => { readyResolve = resolve; });

    const pending = new Map();
    let nextId = 0;

    worker.onmessage = (e) => {
        const { type, id, result, error } = e.data;
        if (type === 'ready') readyResolve();
        if (type === 'result') {
            pending.get(id)?.resolve(result);
            pending.delete(id);
        }
        if (type === 'error') {
            pending.get(id)?.reject(new Error(error));
            pending.delete(id);
        }
    };

    // Kick off initialization
    worker.postMessage({ type: 'init' });

    return {
        ready,
        classify(text) {
            const id = nextId++;
            return new Promise((resolve, reject) => {
                pending.set(id, { resolve, reject });
                worker.postMessage({ type: 'classify', payload: { id, text } });
            });
        },
        terminate() {
            worker.terminate();
        },
    };
}

// Usage:
// const clf = createClassifier();
// await clf.ready;
// const { label, score } = await clf.classify("restart the cluster");
```

---

## Step 5: Hosting the Model Files

The model files need to be served as static assets. Three options:

**A. Bundle with the app.** Put `model.safetensors`, `config.json`, and `tokenizer.json` in your static assets directory. Simplest, but adds ~17MB (for bert-tiny) to your app bundle.

**B. Lazy-load from a CDN.** Host the files on a CDN or your own server. Fetch on first use, cache in the browser (IndexedDB or Cache API). The worker init handles this — first load is slow, subsequent loads are instant.

**C. Host on HuggingFace Hub.** Push your fine-tuned model to a HuggingFace repo, fetch directly from there. Free hosting, versioned, but adds an external dependency.

For the swarm, option B makes the most sense — lazy-load and cache locally. The model files are small enough that the initial download is barely noticeable.

---

## File Checklist

After all steps, you should have:

```
Python side (training):
├── training_data.csv              # labeled examples
├── train.py                       # training script
└── swarm-classifier/              # fine-tuned model output
    ├── model.safetensors          # weights (~17MB for bert-tiny)
    ├── config.json                # architecture config
    ├── tokenizer.json             # tokenizer definition
    └── tokenizer_config.json      # tokenizer metadata

Rust side (inference):
├── Cargo.toml
├── src/
│   └── lib.rs                     # Classifier struct + wasm_bindgen
└── pkg/                           # wasm-pack output
    ├── swarm_classifier_bg.wasm   # compiled WASM (~2-5MB)
    ├── swarm_classifier.js        # JS glue
    └── swarm_classifier.d.ts      # types

JS side (integration):
├── classifier-worker.js           # web worker
├── classifier.js                  # main thread API
└── models/                        # static assets
    ├── model.safetensors
    ├── config.json
    └── tokenizer.json
```

---

## Retraining

When you need to add categories or improve accuracy:

1. Add new labeled examples to `training_data.csv`
2. Rerun `train.py` (adjust `NUM_LABELS` and `LABEL_MAP` if adding categories)
3. Copy the new `model.safetensors` to your static assets
4. Update the `LABELS` const in `lib.rs` if categories changed, rebuild WASM
5. Users get the new model on next load (cache bust with a version hash)

The Rust code and WASM binary only need rebuilding if you change the number of labels. Weight-only updates just swap the safetensors file.

---

## Known Gotchas

**Candle API stability.** Candle is pre-1.0. The BERT model API, VarBuilder signatures, and WASM feature flags may change between versions. Pin your dependency versions.

**Safetensors key names.** HuggingFace's `AutoModelForSequenceClassification` wraps the base model under a `bert.` prefix and adds `classifier.weight` / `classifier.bias` at the top level. If you use a different training setup, the key names in safetensors may differ. Inspect with:

```python
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    for key in f.keys():
        print(key, f.get_tensor(key).shape)
```

**Tokenizer padding.** The `tokenizers` crate in WASM doesn't do dynamic padding across a batch — but we're classifying one input at a time, so this doesn't matter. Just make sure `max_length` in training matches what the tokenizer produces at inference (or use truncation only, no padding, since batch size is always 1).

**WASM memory.** The default WASM memory limit is 256MB. A bert-tiny model + tokenizer fits comfortably. If you step up to DistilBERT (~250MB weights), you may need to increase the WASM memory limit at build time.

**First-load latency.** WASM compilation + model download on first load can take 1-3 seconds. After that, inference is ~2-10ms. Consider showing a loading state or classifying optimistically (default to "general" until the classifier is ready).

---

## Development Strategy: Two Teams in Parallel

This work is split across two Claude Code teams working simultaneously:

**Hive Team** (existing: lead + server-engineer + client-engineer) builds the software — the Candle/WASM inference crate, the web worker, the main thread API, the integration with the swarm client's routing layer. They use an off-the-shelf pre-trained model from HuggingFace to build and test against. The model will be bad at the actual classification task — that's fine. The goal is to get the full pipeline working end-to-end: model loads, tokenizer runs, forward pass executes, classification result comes back, routing acts on it.

**ML Team** (new: lead + dataset-engineer + training-engineer) builds the model — generates training data, writes the training/evaluation scripts, fine-tunes the model, validates accuracy, and delivers the final `model.safetensors` + `config.json` + `tokenizer.json` to the hive team.

### The Interface Contract

Both teams work against the same contract. The classifier consumes three files and exposes one function:

**Inputs (model files):**
- `model.safetensors` — weights, BERT architecture, any size
- `config.json` — HuggingFace model config (hidden_size, num_labels, etc.)
- `tokenizer.json` — HuggingFace tokenizer definition

**Output (classify function):**
```json
{
  "scores": { "general": 0.03, "swarm": 0.97 },
  "label": "swarm",
  "score": 0.97
}
```

The hive team builds everything above and below this contract. The ML team produces the three files. When the fine-tuned model is ready, swapping it in is a file replacement — no code changes, no WASM rebuild (as long as `NUM_LABELS` is read from `config.json`, not hardcoded).

### Off-the-Shelf Model for Development

The hive team needs a model to develop against immediately. Any pre-trained BERT-architecture model with a classification head will work, even if it classifies the wrong thing. Options:

| Model | Params | What it actually classifies | Why it works for dev |
|-------|--------|----------------------------|---------------------|
| `prajjwal1/bert-tiny` + random head | ~4M | Nothing (untrained head) | Smallest, fastest iteration. Outputs are random but the pipeline is real. |
| `distilbert-base-uncased-finetuned-sst-2-english` | ~66M | Sentiment (positive/negative) | Bigger but has a working classification head. Proves the full forward pass works. |
| `typeform/distilbert-base-uncased-mnli` | ~66M | NLI entailment | Can be hacked into zero-shot classification. Closer to useful. |

**Recommendation: start with `prajjwal1/bert-tiny` with a randomly initialized 2-class head.** It's 17MB, loads instantly, and exercises the entire pipeline. The classifications will be garbage — that's expected and irrelevant. The hive team's job is to make the plumbing work, not to make the model smart.

### Timeline

Both teams start at the same time. The hive team will likely finish first (the plumbing is well-defined). The ML team's work is more iterative — dataset quality matters, and validation may require multiple training runs. When the ML team delivers the final model files, the hive team does a file swap and the feature is live with real accuracy.

---

## Roadmap: Observers vs Swarm Users

Classification gates what users can access. The behavior depends on whether the user is an **observer** (connected but not participating) or a **swarm member** (actively joined).

### Observers

Observers get the classifier model loaded locally. When they type something that classifies as swarm-related, they don't get routed to the swarm. Instead, they get a response along the lines of:

> "Join the swarm to know the swarm."

The classifier's job here is to detect swarm-intent and block it, not to route it. The observer experience is: general queries work normally (routed to whatever handles general input), swarm queries get the join prompt.

This means the classifier needs to be good enough to avoid false positives (general queries incorrectly tagged as swarm → user gets a confusing "join the swarm" message) and false negatives (swarm queries slipping through as general → observer accidentally gets swarm info they shouldn't). For observers, **false positives are worse** — they break the general-use experience.

### Swarm Members

Users who have joined the swarm get full classification and routing. Their input is tagged and sent to the appropriate swarm subsystem.

**Tentatively, swarm members will have their input tagged by the swarm itself** — sent to peers for classification rather than classified locally. This is an experiment to test the feasibility of swarm-side tagging:

- How much latency does the network round-trip add?
- Is the classification quality better when using more capable peer models?
- Can the swarm handle the tagging load on top of its other work?

The local classifier still exists as a fallback. If swarm-side tagging proves too slow or unreliable, we drop it and use the local model — by that point, we already know how to do local classification and have a trained model.

The experiment flow:

```
Swarm member types input
       │
       ▼
  Send to swarm for tagging
       │
       ├── Response within threshold (e.g., 500ms) → use swarm tag
       │
       └── Timeout → fall back to local classifier
```

This dual-path approach means we're never blocked on the experiment's outcome. The local classifier is the safe default; swarm tagging is the ambitious upgrade.

### Classification Behavior Matrix

| User Type | Input Tagged As | Behavior |
|-----------|----------------|----------|
| Observer  | general        | Route normally |
| Observer  | swarm          | "Join the swarm to know the swarm" |
| Member    | general        | Route normally |
| Member    | swarm          | Route to swarm (tagged by swarm or local fallback) |

---

## Future: Multi-Class Classification (12 Categories)

The binary swarm/general split is the starting point. The real target is ~12 classes, one per swarm data layer category — routing user input to the right subsystem, not just deciding whether it's swarm-related.

### What Changes

**Training data.** Instead of 300-500 examples split across 2 labels, you need 300-500 *per class*. For 12 classes, that's 3,600-6,000 labeled examples. Generating this synthetically with Claude is still feasible, but requires much more careful prompt design to ensure the classes are well-separated and edge cases are covered.

**Model capacity.** This is where model choice matters significantly:

| Model | 2 classes | 12 classes | Notes |
|-------|-----------|------------|-------|
| `bert-tiny` (4M) | Fine | Risky | May not have enough representational capacity to separate 12 classes cleanly, especially if some are semantically close. |
| `bert-mini` (11M) | Overkill | Good | The sweet spot for 12-class. ~45MB safetensors, still fast in WASM. |
| `bert-small` (29M) | Overkill | Good | More headroom if mini struggles. ~110MB. |
| `distilbert-base` (66M) | Overkill | Comfortable | Reliable but ~250MB starts to strain WASM memory. |

**Recommendation: start with bert-tiny for the binary classifier, but build the pipeline knowing you'll step up to bert-mini for multi-class.** The inference code is identical — same architecture, same forward pass, just different weight dimensions. The only code change is the `LABELS` array and `NUM_LABELS`.

**Class ambiguity.** With 2 classes, the boundary is clear. With 12, some categories will inevitably overlap. A user saying "send a message to node 5" could plausibly be "messaging", "node management", or "routing". This is where the confidence score becomes important — if the top prediction is below a threshold (say 0.7), you can either fall back to a default, ask the user to clarify, or use the top-2 predictions to narrow it down.

### Strategy to Get There

**Phase 1 (now): Binary classifier.** Swarm vs general. bert-tiny. Proves the pipeline, ships the feature.

**Phase 2: Expand to 3-4 high-level categories.** Once the binary model is working, split "swarm" into 2-3 coarse buckets (e.g., "node management", "data/messaging", "system status"). Still bert-tiny, but now you're collecting real user inputs and labeling them against the finer categories. This is where you learn which classes are easy to separate and which are confusing.

**Phase 3: Full 12-class taxonomy.** With real data from Phase 2 and a clear understanding of class boundaries, train the full multi-class model. Step up to bert-mini. This is also where you may want to reconsider the training approach — if some classes are rare, use oversampling or class weights. If some are semantically close, consider a hierarchical approach (first classify into 3-4 groups, then sub-classify).

### Hierarchical Classification (Optional, Later)

If 12-class accuracy is disappointing with a flat classifier, a two-stage approach can help:

```
Input → Stage 1 (coarse, 4 classes) → Stage 2 (fine, 3 classes within the coarse bucket)
```

This requires two models (or one model with two heads), but each individual decision is easier. The WASM pipeline supports this — you'd just load two sets of weights and run two forward passes. Still ~5-20ms total on CPU.

### Impact on Architecture Decisions Now

Even though you're starting binary, a few choices now will save pain later:

- **Use `NUM_LABELS` as a config, not a constant.** Make the Rust code read the number of labels from `config.json` rather than hardcoding it. This way, swapping from 2 to 12 classes is a model file swap, not a code change.
- **Return the full probability distribution**, not just the top label. The `classify()` function should return all scores: `{ "scores": { "general": 0.03, "swarm": 0.97 } }`. This lets the routing layer make its own decisions about thresholds, fallbacks, and ambiguity handling without changing the classifier API.
- **Start collecting real inputs now.** Even while the binary classifier is running, log the inputs and their predicted labels (with user consent). This becomes your training data for the 12-class model. Real data is worth 10x synthetic data for understanding class boundaries.
- **Pick bert-mini as your target architecture.** Fine-tune bert-tiny now for the binary case, but write the Candle inference code against the BERT architecture generically. When you swap to bert-mini later, the code doesn't change — just the weight file gets bigger (~17MB → ~45MB).
