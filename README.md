# 🚀 Flexible-TADA: Adaptive Semantic Recalibration for Encoder-Based Language Models

**Flexible-TADA** is a research codebase for a Parameter-Efficient Fine-Tuning (PEFT) method that combines **input-embedding** updates with **last-transformer-layer** updates, while keeping the entire intermediate backbone frozen. The repository contains the full experimental pipeline used to compare this method against Full Fine-Tuning (FFT), LoRA, and a Static-TADA (embeddings-only) baseline on the **GLUE benchmark**, plus tooling for representation-collapse (CKA) and explainability (Integrated Gradients) analysis.

---

## 💡 Motivation

Purely shallow PEFT methods — i.e. tuning only the input embeddings — often suffer from **Representation Collapse**: the adapted signal has to propagate through a fully frozen, deep Transformer stack, which limits how well the model can re-align its representations with a new downstream task.

**Flexible-TADA** addresses this by selectively unfreezing two specific parts of the network:
1. The **input embedding layer** — adapts the model's input-level semantic space.
2. The **final Transformer block** — gives the model one trainable "semantic safety net" close to the output, so high-level task-specific representations can be re-projected correctly.

Everything in between stays frozen, preserving the pre-trained reasoning backbone while still allowing meaningful adaptation — with **zero added parameters and zero inference overhead**, unlike adapter-style methods.

### ✨ Key Properties
- ⚡ **Multi-architecture support** — configs are provided for `BERT`, `RoBERTa`, `DeBERTa-v3-base`, `DeBERTa-v2-xxlarge`, `ELECTRA`, and the decoder-only `Qwen2-0.5B` / `Qwen3.5-2B` (used for architectural-validity comparisons against deep encoders).
- 🧠 **Built-in explainability tooling** — Integrated Gradients (via [Captum](https://captum.ai/)) based Faithfulness and Sufficiency metrics, plus token-attribution heatmaps.
- 📐 **Representation-collapse diagnostics** — layer-wise linear CKA comparison against an FFT reference model.
- ⏱️ **Zero inference overhead** — Flexible-TADA only toggles `requires_grad` on existing weights; it adds no new modules, so inference cost is identical to the base model.
- 🌵 **Few-shot evaluation support** — seeded sub-sampling utilities for low-resource ("data drought") robustness experiments, with reproducible per-seed sampling.

> **Scope note:** This codebase targets **GLUE only**. There is no SuperGLUE support — task routing, tokenization, and metrics are all scoped strictly to the 8 GLUE tasks (`mnli`, `sst2`, `mrpc`, `cola`, `qnli`, `qqp`, `rte`, `stsb`).

---

## 📂 Repository Structure

```text
Flexible-TADA/
├── configs/                          # YAML configs (one per base architecture)
│   ├── bert_glue.yaml
│   ├── roberta_glue.yaml
│   ├── deberta_glue.yaml             # DeBERTa-v3-base
│   ├── deberta-v2-xxlarge_glue.yaml
│   ├── electra_glue.yaml
│   ├── Qwen2-0.5B_glue.yaml
│   └── Qwen3.5-2B_glue.yaml
│
├── data/
│   ├── dataset_builder.py            # Loads GLUE tasks, seeded few-shot sampling
│   └── data_utils.py                 # Tokenization, dynamic padding, data collator
│
├── models/
│   ├── model_factory.py              # Builds the base model and routes to the chosen method
│   ├── flex_tada.py                  # Our method: unfreezes embeddings + last layer
│   ├── baselines.py                  # Static-TADA baseline: unfreezes embeddings only
│   └── __init__.py
│
├── trainer/
│   ├── engine.py                     # Builds the HuggingFace Trainer (per-method LR routing, etc.)
│   ├── evaluator.py                  # Final evaluation + inference-latency measurement
│   └── xai_evaluator.py              # Integrated-Gradients based XAI evaluation loop
│
├── utils/
│   ├── metrics.py                    # Per-task GLUE metrics (accuracy, F1, MCC, Pearson/Spearman)
│   ├── xai_metrics.py                # Faithfulness / Sufficiency score computation
│   ├── cka_metrics.py                # Linear CKA similarity for representation analysis
│   ├── memory_profiler.py            # Peak-VRAM and latency profiling utilities
│   └── logger.py                     # Console + file logging setup
│
├── scripts/                          # Bash entry points for every experiment
│   ├── run_glue.sh                   # Full GLUE sweep across 4 encoders × 4 methods × 8 tasks
│   ├── run_ablation.sh               # Layer-depth ablation (unfreeze layer 0 / 6 / 10 / 11)
│   ├── run_fewshot.sh                # Few-shot robustness sweep (10/50/100 samples × 3 seeds)
│   ├── run_xai.sh                    # Faithfulness/Sufficiency evaluation on saved checkpoints
│   ├── run_cka_analysis.sh           # Wrapper for run_representation_analysis.py
│   ├── run_stacked_heatmap.sh        # Wrapper for generate_stacked_heatmap.py
│   └── run_validity_analysis.sh      # Encoder-depth vs. decoder causal-masking comparison
│
├── main.py                           # Main training + evaluation entry point
├── run_xai_analysis.py               # Standalone XAI evaluation on a single trained checkpoint
├── run_representation_analysis.py    # Layer-wise CKA: FFT vs. Static-TADA vs. Flex-TADA
├── generate_stacked_heatmap.py       # Generates multi-method IG attribution heatmaps for fixed examples
└── requirements.txt                  # Python dependencies
```

---

## ⚙️ Installation

```bash
git clone https://github.com/sepehrmustafavi/Flexible-TADA.git
cd Flexible-TADA

# Python 3.10+ recommended
pip install -r requirements.txt
```

Notes on dependencies:
- `transformers>=4.46.0` is a **hard requirement**, not just a suggestion — `trainer/engine.py` uses `eval_strategy` and `processing_class`, both of which only exist from that version onward.
- `sentencepiece` support is required for DeBERTa-v2/v3 tokenizers (pulled in transitively by `transformers`).
- `flash-attn` and `deepspeed` are listed for large-model / multi-GPU runs (e.g. Qwen3.5-2B, DeBERTa-v2-xxlarge); they are optional if you only run the smaller encoder configs (BERT/RoBERTa/ELECTRA/DeBERTa-v3-base).
- `captum`, `matplotlib`, and `seaborn` are only needed for the XAI / heatmap scripts.

---

## 🧩 Configuration System

Every experiment is driven by a single YAML file under `configs/`. Each config has four sections:

```yaml
model:        # model_name_or_path, tokenizer, max_seq_length, trust_remote_code
dataset:      # benchmark (glue), task list, validation split name
training:     # batch size, epochs, scheduler, eval/save strategy, etc.
methods:      # one block per tuning method, each with its own learning_rate
  fft: {...}
  lora: {...}             # r, lora_alpha, lora_dropout, target_modules
  static_tada: {...}      # trainable_layers (embedding keyword)
  flex_tada: {...}        # trainable_layers (embedding keyword + last-layer keyword)
system:       # seed, fp16/bf16, logging backend
```

`flex_tada.trainable_layers` accepts either an explicit layer-name keyword (e.g. `"encoder.layer.11"`, `"model.layers.27"`) or the literal string `"last_layer"`, which is **resolved dynamically at runtime** from the model's `num_hidden_layers` / `n_layers` config, with architecture-specific naming for BERT/RoBERTa/ELECTRA, DeBERTa, and Qwen/LLaMA-style models (see `models/flex_tada.py`). The resolution is done on a local copy of the layer list, so it's safe to call the model factory repeatedly on the same loaded config (as the CKA and heatmap scripts do) without side effects.

---

## 🚀 Quickstart: Single Run

```bash
python main.py \
    --config configs/roberta_glue.yaml \
    --method flex_tada \
    --task sst2 \
    --seed 42
```

Supported `--method` values: `fft`, `lora`, `static_tada`, `flex_tada`.

Optional flags:
- `--task <name>` — overrides the task list in the YAML and runs a single GLUE task.
- `--seed <int>` — overrides the YAML seed. This now correctly propagates to **both** the training seed and the few-shot sampling seed.
- `--few_shot <int>` — sub-samples the training set to N examples (seeded, reproducible per `--seed`).

Each run writes to `outputs/<task>_<method>_<seed>/`, including HuggingFace Trainer checkpoints, a `logs/` directory, and a `results_<task>_<method>.json` file with GLUE metrics, training throughput, and per-sample inference latency.

---

## 🔁 Reproducing the Full Experiment Suite

Make the scripts executable first:
```bash
chmod +x scripts/*.sh
```

1. **Full GLUE benchmark across encoders**
   Sweeps `BERT`, `RoBERTa`, `DeBERTa-v3-base`, and `ELECTRA` × `{fft, static_tada, lora, flex_tada}` × all 8 GLUE tasks.
   ```bash
   ./scripts/run_glue.sh
   ```

2. **Layer-depth ablation**
   Re-runs Flexible-TADA on SST-2 while sweeping which transformer layer is unfrozen (0, 6, 10, 11), to isolate the contribution of the final layer.
   ```bash
   ./scripts/run_ablation.sh
   ```

3. **Few-shot robustness ("data drought")**
   Evaluates `MNLI` and `RTE` at 10/50/100 training samples across 3 seeds (42, 123, 2026) for all four methods. Each seed now produces a genuinely different training subset as well as a different model initialization.
   ```bash
   ./scripts/run_fewshot.sh
   ```

4. **Semantic robustness (XAI: Faithfulness & Sufficiency)**
   Locates the best checkpoint for each method (matched by epoch from `results_*.json` / `trainer_state.json`) and runs Integrated-Gradients-based evaluation on the full validation set. Loads checkpoints saved either as `.bin` or `.safetensors`, and infers `num_labels` per task.
   ```bash
   ./scripts/run_xai.sh
   ```

5. **Representation-collapse analysis (Linear CKA)**
   Compares layer-wise hidden-state similarity of Static-TADA and Flex-TADA against an FFT reference model on SST-2.
   ```bash
   ./scripts/run_cka_analysis.sh
   ```

6. **Stacked attribution heatmaps**
   Generates side-by-side Integrated-Gradients heatmaps (FFT / Static-TADA / Flex-TADA / LoRA) for a fixed set of illustrative SST-2 and MRPC examples — used for the paper's qualitative figures. Loads checkpoints saved either as `.bin` or `.safetensors`, and warns explicitly if neither is found for a given method.
   ```bash
   ./scripts/run_stacked_heatmap.sh
   ```

7. **Architectural validity analysis (Encoder depth vs. Causal masking)**
   Compares the very deep `DeBERTa-v2-xxlarge` encoder against `Qwen2-0.5B` / `Qwen3.5-2B` decoders on `sst2`/`qnli`, driven by their respective YAML configs. Per-model results are organized under `outputs/<config_name>/`.
   ```bash
   ./scripts/run_validity_analysis.sh
   ```

---

## 📊 Output Artifacts

| Path | Contents |
|---|---|
| `outputs/<task>_<method>_<seed>/results_*.json` | Final GLUE metric(s), train runtime, eval latency per sample |
| `outputs/few_shot_results/<trial_name>/` | Same as above, organized per few-shot trial |
| `outputs/xai_results/xai_<task>_<method>.json` | Mean/std Faithfulness and Sufficiency scores |
| `outputs/xai_results/heatmap_<task>_<method>.png` | Single-method token-attribution heatmap |
| `outputs/representation_analysis/cka_<task>.json` | Layer-wise CKA scores (FFT vs. Static-TADA, FFT vs. Flex-TADA) |
| `outputs/visualizations/<case_name>.png` | Multi-method stacked attribution heatmap for a fixed example |
| `outputs/<config_name>/<task>_<method>_<seed>/` | Validity-analysis results, grouped per model config |

---
