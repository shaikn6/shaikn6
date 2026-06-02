# On-Device LLM Optimizer — Design Spec

**Date:** 2026-06-01  
**Project:** `on-device-llm-optimizer`  
**Status:** Approved

---

## 1. Problem

Interview question: "Design an on-device LLM that runs fully offline on an iPhone."

Most candidates answer conceptually. This project builds the actual pipeline — proving distillation, quantization, and CoreML export with real benchmark numbers on Apple Silicon.

---

## 2. Goal

Distill Phi-3 Mini 3.8B → 1B student model on Mac (MLX). Quantize to INT4. Export to CoreML. Benchmark all variants. Show results in Streamlit dashboard.

**Resume bullet target:**
```
Implemented knowledge distillation pipeline on Apple Silicon (MLX)
compressing Phi-3 Mini 3.8B → 1B student model; achieved ~87% MMLU
retention at 7.6× size reduction (2.2GB → 500MB INT4) with CoreML
export targeting iPhone Neural Engine.
```

---

## 3. Architecture

```
┌─────────────────────────────────────────────┐
│         Teacher: Phi-3 Mini 3.8B            │
│         (MLX INT4, ~2.2GB, frozen)          │
└──────────────────┬──────────────────────────┘
                   │ soft labels (T=4)
                   ▼
┌─────────────────────────────────────────────┐
│    Knowledge Distillation (MLX, Alpaca 52K) │
│    Loss = 0.7 × KL_soft + 0.3 × CE_hard    │
│    Checkpoint every 500 steps               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│      Student: Custom 1B Transformer         │
│      (12 layers, 1024 dim, 8 heads)         │
│      MLX-native, ~3.9GB float32             │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
┌──────────────────┐  ┌───────────────────────┐
│ MLX INT4 Quant   │  │  CoreML Export        │
│ ~500MB           │  │  .mlpackage           │
│ Group size: 64   │  │  Target: iPhone NE    │
└──────────┬───────┘  └──────────┬────────────┘
           │                     │
           └──────────┬──────────┘
                      ▼
┌─────────────────────────────────────────────┐
│         Streamlit Benchmark Dashboard       │
│  Tab 1: Live inference (prompt comparison)  │
│  Tab 2: Metrics (size/speed/MMLU/perplexity)│
│  Tab 3: Architecture + resume story         │
└─────────────────────────────────────────────┘
```

---

## 4. Approach

**Approach A — Full pipeline: distillation → quantization → CoreML**

- MLX framework throughout (Apple Silicon native)
- Teacher: Phi-3 Mini 3.8B loaded in INT4 (frozen, memory-efficient)
- Student: custom 1B transformer built in `mlx.nn`
- Distillation: sequence-level KD, soft labels + hard labels
- Post-distill: MLX INT4 quantization
- Export: `coremltools` → `.mlpackage` targeting CPU+NE
- Compute: local Mac M-series, ~6-10 hours training

---

## 5. Benchmark Targets

| Variant | Size | Tokens/sec | MMLU |
|---------|------|-----------|------|
| Teacher (Phi-3 INT4) | 2.2GB | baseline | 68.8 |
| Student FP32 | ~3.9GB | slower | ~55-60 |
| Student INT4 | ~500MB | faster | ~52-57 |
| Student CoreML | ~500MB | fastest (NE) | ~52-57 |

---

## 6. Technical Decisions

| Concern | Decision | Reason |
|---------|----------|--------|
| Framework | MLX | Apple Silicon native, unified memory, no CUDA needed |
| Teacher | Phi-3 Mini 3.8B INT4 | High quality, fits in 8GB RAM, free |
| Student arch | 12L × 1024d × 8h | 1B params, same tokenizer as teacher |
| KD temperature | 4.0 | Softens distributions, better knowledge transfer |
| KD alpha | 0.7 soft / 0.3 hard | Empirically strong for instruction-following |
| Dataset | Alpaca 52K | Clean, instruction-following, fits RAM |
| Quantization | INT4, group_size=64 | Balances speed vs quality |
| CoreML target | CPU_AND_NE | Activates Neural Engine for matrix ops |
| Dashboard | Streamlit | Fast, visual, no JS needed |

---

## 7. File Structure

```
on-device-llm-optimizer/
├── src/
│   ├── model/
│   │   ├── student.py          # 1B transformer (MLX nn.Module)
│   │   └── config.py           # hyperparams: layers, dim, heads, vocab
│   ├── distillation/
│   │   ├── trainer.py          # KD training loop + checkpointing
│   │   ├── dataset.py          # Alpaca 52K loader + tokenization
│   │   └── losses.py           # KL divergence (soft) + CE (hard)
│   ├── optimization/
│   │   ├── quantize.py         # MLX INT4 (group_size=64)
│   │   └── memory.py           # RAM profiler for benchmark
│   ├── export/
│   │   └── coreml_export.py    # coremltools → .mlpackage (CPU+NE)
│   ├── evaluation/
│   │   ├── perplexity.py       # held-out set perplexity
│   │   └── mmlu_eval.py        # MMLU 5-shot, 200 questions
│   └── app/
│       └── streamlit_app.py    # 3-tab dashboard
├── configs/
│   └── distill_config.yaml     # single source of truth
├── scripts/
│   ├── download_data.py        # fetch Alpaca 52K
│   ├── train.py                # orchestrate distillation
│   └── export.py               # quantize + CoreML export
├── checkpoints/                # saved every 500 steps
├── models/
│   ├── student_fp32/
│   ├── student_int4/
│   └── student.mlpackage
├── tests/
│   ├── test_student_model.py
│   ├── test_losses.py
│   ├── test_quantize.py
│   └── test_benchmark.py
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 8. Configuration (`distill_config.yaml`)

```yaml
teacher:
  model: microsoft/Phi-3-mini-4k-instruct
  quantize: int4

student:
  layers: 12
  hidden_dim: 1024
  attention_heads: 8
  vocab_size: 32064

distillation:
  temperature: 4.0
  alpha: 0.7
  dataset: tatsu-lab/alpaca
  max_samples: 52000
  batch_size: 8
  steps: 10000
  checkpoint_every: 500
  lr: 3e-4

quantization:
  bits: 4
  group_size: 64

export:
  target: CoreML
  compute_units: CPU_AND_NE
```

---

## 9. Data Flow

**Training (`scripts/train.py`):**
```
download_data.py
→ Alpaca 52K → tokenize (Phi-3 tokenizer) → 95/5 train/val split

trainer.py per batch:
  teacher.forward(tokens) → logits_T  [frozen, no grad]
  student.forward(tokens) → logits_S
  soft_loss = KL(softmax(logits_S/T), softmax(logits_T/T))
  hard_loss = CE(logits_S, labels)
  loss = 0.7 × soft_loss + 0.3 × hard_loss
  loss.backward() → optimizer.step()
  every 500 steps → save checkpoint
```

**Export (`scripts/export.py`):**
```
load student_fp32
→ quantize.py (MLX INT4, group=64) → models/student_int4/
→ coreml_export.py → models/student.mlpackage
```

**Benchmark:**
```
for variant in [teacher, student_fp32, student_int4, student_coreml]:
  measure: disk size, peak RAM, tokens/sec, perplexity, MMLU@200
→ results.json → Streamlit charts
```

---

## 10. Streamlit Dashboard

**Tab 1 — Live inference:**
- Prompt input → runs all 4 variants → side-by-side output + speed

**Tab 2 — Benchmark charts:**
- Bar: model size (MB)
- Bar: tokens/sec
- Line: MMLU score vs compression level
- Table: full metrics grid

**Tab 3 — Architecture + resume story:**
- Pipeline diagram (static PNG)
- Maps to interview answer steps 1-7
- Resume bullet generated from real benchmark numbers

---

## 11. Testing

| Test | Type | Assertion |
|------|------|-----------|
| `test_student_model` | Unit | forward pass → shape `[batch, seq, vocab_size]` |
| `test_losses` | Unit | α=1.0 → pure KL; α=0.0 → pure CE |
| `test_quantize` | Unit | INT4 model ≤ 15% of FP32 size |
| `test_benchmark` | Integration | all 4 variants return non-empty output |

---

## 12. Dependencies

```toml
[project]
dependencies = [
  "mlx",
  "mlx-lm",
  "coremltools",
  "transformers",
  "datasets",
  "streamlit",
  "plotly",
  "psutil",
  "pyyaml",
  "pytest",
]
```

---

## 13. Out of Scope

- Actual iOS Swift app (Python pipeline only)
- Fine-tuning on legal/domain data (general instruction-following only)
- Pruning (quantization + distillation sufficient for scope)
- Multi-GPU / cloud training
- RLHF / preference alignment
