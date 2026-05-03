# 🚀 Flexible-TADA: Adaptive Semantic Recalibration for Encoders

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9D371.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Official PyTorch Implementation** of the paper: *"Flexible TADA: Adaptive Semantic Recalibration for Explainable NLU in Encoder Models"* (Under Review).

## 💡 Overview

Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA and Static TADA have revolutionized the adaptation of pre-trained models. However, purely shallow methods (tuning only embeddings) often suffer from representation collapse when faced with complex Natural Language Understanding (NLU) and semantic disambiguation tasks.

**Flexible-TADA** solves this by strategically unfreezing the **Input Embeddings** AND the **Final Transformer Layer**. This dual-unfreezing acts as a "semantic safety net," allowing the model to project adapted embeddings into the correct high-level task space while keeping the vast majority of the backbone frozen.

### Key Advantages:
- ⚡ **Universal Encoder Compatibility:** Robustly tested and proven across diverse architectures including `BERT`, `RoBERTa`, `DeBERTa-v3`, and `ELECTRA`.
- 🧠 **Explainable by Design:** Includes post-hoc XAI analysis (Faithfulness & Sufficiency) to prove deep semantic reliance rather than surface-level heuristics.
- 🌵 **Data-Drought Resistant:** Superior robustness in Few-Shot scenarios compared to standard FFT and LoRA.
- ⏱️ **Green AI:** Faster inference latency compared to LoRA, avoiding complex multi-matrix sequential operations during forward passes.

---

## 📂 Repository Structure

```text
Flexible-TADA/
├── configs/                # YAML configs for different architectures
│   ├── bert_glue.yaml
│   ├── deberta_glue.yaml
│   ├── electra_glue.yaml
│   └── roberta_glue.yaml
├── data/                   # Dynamic dataset builders for GLUE benchmark
├── models/                 # Core architecture (Flex-TADA, LoRA, Baselines)
├── scripts/                # Bash scripts for 1-click reproducibility
│   ├── run_table1_glue.sh  # Massive evaluation across 4 models
│   ├── run_ablation.sh
│   ├── run_fewshot.sh
│   └── run_xai_table5.sh   # XAI automated analysis
├── trainer/                # Execution engines
│   ├── engine.py           # HF Trainer wrapper
│   ├── evaluator.py        # Latency & performance evaluation
│   └── xai_evaluator.py    # Post-hoc semantic evaluator
├── utils/                  # Core utilities
│   ├── logger.py           # Logging configuration
│   ├── memory_profiler.py  # Hardware VRAM/Latency tracking
│   ├── metrics.py          # Task-specific scoring (F1, MCC, Pearson, etc.)
│   └── xai_metrics.py      # Faithfulness & Sufficiency math
├── main.py                 # The main training/evaluation entry point
├── run_xai_analysis.py     # Post-hoc XAI execution entry point
└── requirements.txt        # Environment dependencies (Tensorboard, DeepSpeed, etc.)
```

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sepehrmustafavi/Flexible-TADA.git](https://github.com/sepehrmustafavi/Flexible-TADA.git)
   cd Flexible-TADA
2. **Create a virtual environment and install dependencies:**
   ```bash
   # We recommend using Python 3.10+
   # Note: sentencepiece is included for DeBERTa-v3 tokenization support
   pip install -r requirements.txt
   ```

## 🚀 Reproducing the Paper Experiments
All experiments from the paper are fully automated using the provided bash scripts. Before running, ensure the scripts have execution permissions:
   ```bash
   chmod +x scripts/*.sh
   ```

1. **Massive GLUE Benchmark across Encoders**
Evaluates FFT, LoRA, Static TADA, and Flex TADA across the entire GLUE benchmark. It automatically iterates through BERT, RoBERTa, DeBERTa-v3, and ELECTRA configurations.
    ```bash
   ./scripts/run_table1_glue.sh
    ```
2. **Deep Layer Ablation Study**
Automatically modifies the architecture to unfreeze different depths (Layers 0, 6, 10, 11) to empirically prove the critical role of the final layer in semantic extraction.
   ```bash
   ./scripts/run_ablation.sh
   ```
3. **Few-Shot Robustness (Data Drought)**
Evaluates the model's resistance to overfitting in low-data regimes (10, 50, 100 samples) across multiple random seeds for statistical significance.
   ```bash
   ./scripts/run_fewshot.sh
   ```
4. **Semantic Robustness (XAI Analysis)**
Runs post-hoc Explainable AI metrics (Faithfulness and Sufficiency) on the best-saved checkpoints to prove semantic alignment and robustness.
   ```bash
   ./scripts/run_xai_table5.sh
   ```