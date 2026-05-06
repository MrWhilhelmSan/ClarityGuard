# ClarityGuard — Gemma 4 E4B Fine-Tuning for Neurodivergent Communication Support

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) [![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Model-yellow)](https://huggingface.co/CharlieBonito/clarity-guard-gemma4-7b)

**ClarityGuard** is a fine-tuned [Gemma 4 E4B (4B)](https://ai.google.dev/gemma) multimodal model that detects and analyzes ambiguous or problematic communication patterns in workplace conversations. It is designed to help neurodivergent individuals (especially autistic adults) identify manipulation, gaslighting, and unclear communication — giving them a structured framework to respond.

Built for the **Gemma 4 Good Hackathon** (Kaggle, 2026).

**Tracks entered:**
- Main Track: Safety & Trust
- Special Technology Track: Unsloth

---

## Architecture Overview

```
User Message / Image
       │
       ▼
┌─────────────────────────────────────────────┐
│           DIFY Self-Hosted (Orchestrator)    │
│  ┌───────────────────────────────────────┐  │
│  │ RAG Pipeline                          │  │
│  │ Jina Embeddings → Knowledge Base      │  │
│  │ Docs: Chatty 231051 + Author's book  │  │
│  │ on manipulation (10 stages)           │  │
│  │ Prompt: ClarityGuard v4.4             │  │
│  └───────────────────────────────────────┘  │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  ClarityGuard — Gemma 4 E4B (Fine-tuned)    │
│  QLoRA 4-bit | Checkpoint-375              │
│  C.F.R.V.A. Analysis Framework             │
│  Response: 4-step structured analysis      │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
        llama-server --mmproj (port 8081)
        Vision-capable | 100% Local | ~5.3 GB VRAM
```

## How Gemma 4 Was Used

### 1. Teacher Model: Gemma 4 31B (via Ollama + Dify + RAG)

Gemma 4 31B served as the **teacher model** to generate structured training data:

- A publicly available Kaggle dataset of **10,000 manipulation-related conversations** was sourced, covering **7 manipulation types** (gaslighting, guilt-tripping, love-bombing, charm/flattery, direct coercion, passive-aggressive, and neutral) across **4 relationship contexts** (coworker, friend, romantic, family)
- **2,444 coworker conversations** were prioritized as the primary domain, supplemented by ~556 conversations from other contexts (neutral + manipulative), for a curated pool of **3,000 conversations**
- Each conversation was sent to **Gemma 4 31B** (cloud, via Ollama) through Dify (self-hosted, localhost), with RAG context from the Chatty system (231051) and the author's book on manipulation (10 stages), using the ClarityGuard v4.4 system prompt
- After filtering API-error responses, **2,999 structured C.F.R.V.A. analyses** were curated for fine-tuning
- An additional **56 real-world workplace scenarios** (Kaggle Answers) were processed through the same Dify/RAG pipeline for evaluation

### 2. Student Model: Gemma 4 E4B (Fine-tuned with Unsloth)

Gemma 4 E4B (4B parameters) was chosen as the **student model** because:

- **Multimodal**: native vision capabilities via `Gemma4ForConditionalGeneration`
- **Edge-deployable**: runs on consumer GPUs (RTX 4060, ~5.3 GB VRAM)
- **Local-first**: no cloud dependency, privacy-preserving
- Initial attempt with Gemma 4 E2B (2B) was insufficient — the 4B model has the capacity to learn the structured 4-step analysis format

**Fine-tuning technique:** QLoRA 4-bit via [Unsloth](https://unsloth.ai/)

| Hyperparameter | Value |
|---|---|
| LoRA r | 48 |
| LoRA alpha | 96 |
| LoRA dropout | 0.0 |
| Load in 4-bit | True |
| Max sequence length | 4096 |
| Per device batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Learning rate | 1.5e-4 |
| Scheduler | Linear warmup + cosine decay |
| Optimizer | adamw_8bit |
| Epochs | 1 |
| Precision | bf16 |
| Gradient checkpointing | True |

### 3. Training Results

| Metric | Value |
|---|---|
| Initial loss | 9.49 |
| Final loss | 0.72 |
| Minimum loss | 0.64 (step 364) |
| Loss reduction | 92.4% |
| Total steps | 375 |
| Samples seen | 2,999 |

No signs of overfitting — loss decreased steadily throughout training.

### 4. Export & Deployment

The fine-tuned model was exported to GGUF format (Q4_K_M quantization) and deployed with **llama.cpp**:

```bash
./llama-server \
  -m Checkpoint-375-Ollama-Clean-7.5B-Q4_K_M.gguf \
  --mmproj mmproj-Checkpoint-375-Ollama-Clean-BF16.gguf \
  --host 0.0.0.0 --port 8081 \
  -c 16384 -ngl 99 --jinja
```

**Important:** The model works with full vision capabilities in llama.cpp. Exporting to Ollama **loses vision support**.

## The C.F.R.V.A. Framework

The C.F.R.V.A. model is the analytical engine behind ClarityGuard. It was created by Carlos Lengemann (2026) and is published under CC BY 4.0.

| Dimension | Meaning | Score 0-10 |
|---|---|---|
| **C**ontexto no declarado | Hidden assumptions, unstated background | 0-10 |
| **F**ocalización difusa | No measurable criteria for success | 0-10 |
| **R**edirección encubierta | Topic shift without signaling | 0-10 |
| **V**alidación condicionada | Approval depends on not asking questions | 0-10 |
| **A**mbigüedad lingüística | Jargon, metaphors, vague language | 0-10 |

**Total:** /50. Higher scores = more ambiguity.

**Response modes:**
- **0-10:** Clear. Confirm briefly.
- **11-20:** Minor ambiguity. Suggest one question.
- **21-30:** Moderate. Full analysis + clarification.
- **31-50:** Maximum alert. 4-step response with cognitive protection.

## Response Format

Every ClarityGuard analysis follows 4 structured steps:

```
STEP 1 — ANALYSIS
🔍 [ClarityGuard] C.F.R.V.A. score: XX/50 → [level]
[Structural analysis of the message]

STEP 2 — COGNITIVE PROTECTION (if score ≥ 21)
🔒 Your confusion is not a failure. It is the correct response
   to an incomplete message.

STEP 3 — CONCRETE ACTION (Read-Back)
✍️ [Specific clarification question the user can send]

STEP 4 — FOLLOW-UP PLAN (if score ≥ 31)
⏰ [Strategy if clarification is still abstract]
```

## RAG System

ClarityGuard uses Retrieval-Augmented Generation with two proprietary documents:

1. **Chatty 231051** — System identity and symbolic companion framework
2. **Author's Book on Manipulation** — 10 stages of manipulation, personal experience in corporate environments

Both documents are indexed via Jina Embeddings and queried by Dify during teacher-model inference. The ClarityGuard v4.4 system prompt instructs the model to use RAG context for real-world ambiguity examples and structural communication analysis.

**Stack:** Dify (self-hosted) + Jina Embeddings

## Dataset Pipeline

```
Kaggle dataset (10,000 manipulation conversations)
  7 types: gaslighting, guilt-tripping, love-bombing,
  charm/flattery, direct coercion, passive-aggressive, neutral
  4 contexts: coworker, friend, romantic, family
        │
        ▼
filter_dataset.py → 2,444 coworker + ~556 other = 3,000 curated
        │
        ▼
Dify batch scripts (3 × 1,000) → Gemma 4 31B (teacher) + RAG
  RAG: Chatty 231051 + Author's manipulation book (10 stages)
  Prompt: ClarityGuard v4.4
  → 2,999 structured C.F.R.V.A. responses
        │
        ▼
build_training_dataset.py → Gemma 4 chat template format
        │
        ▼
build_clean_gemma4_unsloth.py → Clean: forbidden terms, language normalization
        │
        ▼
train_clarityguard_sft.py → Unsloth QLoRA on Gemma 4 E4B
```

## Hardware Requirements

### Training (Unsloth + QLoRA)
| Component | Spec |
|---|---|
| CPU | Intel Core i5 13600KF |
| RAM | 32 GB DDR5 5600 MHz |
| GPU | RTX 5070 Ti (16 GB VRAM) |
| Storage | NVMe PCIe 4.0 |

### Inference (llama.cpp + Vision)
| Component | Spec |
|---|---|
| CPU | Intel Core i5 11400 |
| RAM | 32 GB DDR4 3200 MHz |
| GPU | RTX 4060 |
| VRAM usage | ~5.3 GB (with vision) |
| Storage | NVMe 500 GB |

## Repository Structure

```
ClarityGuard/
├── README.md                          ← This file
├── LICENSE                            ← CC BY 4.0
├── requirements.txt
│
├── training/
│   ├── train_clarityguard_sft.py      ← Main Unsloth training script
│   ├── train_config.yaml              ← Training hyperparameters
│   ├── prepare_dataset.py             ← Dataset preparation
│   └── normalize_assistant_language.py← Language normalization
│
├── data/
│   ├── build_gemma4_e2b_final.py      ← Clean + forbidden terms replacement
│   ├── build_gemma4_sft_internalize.py← Internalize think blocks
│   └── filter_dataset.py              ← Filter 10K → 3K curated
│
├── dify/
│   ├── dify_batch_chat.py             ← Main Dify batch engine (SSE client)
│   ├── dify_batch_chat{1..9000}.py   ← Batch wrappers (9 parts)
│   ├── dify_batch_kaggle.py           ← Evaluate with real autism cases
│   ├── build_training_merge_dify_outputs.py  ← Merge Dify responses
│   ├── build_clean_gemma4_unsloth.py  ← Build Unsloth format
│   ├── build_clean_gemma4_chat1_5000.py ← Extended merge
│   └── informe_progreso_batch_dify.txt ← Batch progress report
│
├── export/
│   └── exportar.py                    ← Export checkpoint to HF/GGUF
│
├── deploy/
│   ├── run-llama-server-q4.sh         ← llama-server launcher (Q4)
│   ├── run-llama-server.sh            ← llama-server launcher (BF16)
│   ├── clarity-e4b-vision.service     ← systemd service (auto-start)
│   ├── Dockerfile                     ← Container deployment
│   ├── docker-compose.yml             ← Docker compose config
│   └── modal_deploy.py               ← Cloud deployment script
│
├── prompts/
│   └── clarityguard_prompt_v4.4.txt   ← Full system prompt
│
├── docs/
│   └── manifiesto_cfrva_v1.0.md       ← C.F.R.V.A. model definition
│
├── Documentation/
│   ├── manipulational_conversation.jsonl  ← Original 10K Kaggle dataset
│   ├── manipulational_conversation_with_model_responses_train_ready.jsonl ← 2,999 curated + teacher responses
│   ├── manipulational_conversation_with_model_responses_train_ready_unsloth_format.jsonl ← Unsloth format
│   ├── manipulational_conversation_responses_all.txt  ← Consolidated teacher responses (5,634 valid)
│   ├── Kaggle Answers.xlsx              ← 56 real-world workplace scenarios + C.F.R.V.A. analyses
│   ├── promt 4.4.docx                  ← ClarityGuard v4.4 system prompt (Dify)
│   └── README_RESULTADOS_DATASET.txt    ← Dataset pipeline summary
│
└── config/
    └── (inference configuration examples)
```

## Quick Start

### 1. Serve the Model (llama.cpp)

```bash
# Download the model weights from HuggingFace:
# https://huggingface.co/CharlieBonito/clarity-guard-gemma4-7b

# Or clone the repo and deploy:
git clone https://github.com/MrWhilhelmSan/ClarityGuard.git
cd ClarityGuard/deploy
chmod +x run-llama-server-q4.sh
./run-llama-server-q4.sh
# Server starts at http://localhost:8081
```

### 2. Test the API

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clarityguard",
    "messages": [
      {"role": "user", "content": "They said: \"We need to fix this.\" What does that mean?"}
    ]
  }'
```

### 3. Fine-tune Your Own (Unsloth)

```bash
pip install unsloth
cd training
python train_clarityguard_sft.py --config train_config.yaml --profile pilot_1k
```

## License

- **Code:** Apache 2.0
- **C.F.R.V.A. Model Definition:** CC BY 4.0 (attribution: Carlos Lengemann, 2026)
- **RAG Documents (Chatty + Book):** All rights reserved by Carlos Lengemann

---

**Competition:** [Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good-hackathon) (2026)
**Author:** Carlos Lengemann
**GitHub:** https://github.com/MrWhilhelmSan/ClarityGuard
**Model Weights:** https://huggingface.co/CharlieBonito/clarity-guard-gemma4-7b
