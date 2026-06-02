# ClarityGuard - Gemma 4 E4B for Neuro-Inclusive Communication Clarity

[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/CharlieBonito/clarity-guard-gemma4-7b)
[![Live Demo](https://img.shields.io/badge/HuggingFace-Space-blue)](https://huggingface.co/spaces/CharlieBonito/ClarityGuardAgent)
[![Framework License: CC BY 4.0](https://img.shields.io/badge/C.F.R.V.A.-CC_BY_4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

ClarityGuard is a fine-tuned Gemma 4 E4B IT model for structural communication analysis. It helps neurodivergent users decode ambiguous workplace and personal messages by identifying what a message says, what it leaves undefined, and what clarification can be sent back.

The core principle is simple: when a message lacks a clear owner, action, deadline, or measurable criterion, confusion is a valid response to incomplete input, not a cognitive error.

Built for the Gemma 4 Good Hackathon 2026.

## Links

| Resource | URL |
|---|---|
| Live demo | https://huggingface.co/spaces/CharlieBonito/ClarityGuardAgent |
| Model weights | https://huggingface.co/CharlieBonito/clarity-guard-gemma4-7b |
| Training dataset | https://huggingface.co/datasets/CharlieBonito/clarity-guard-training-data |
| Source repository | https://github.com/MrWhilhelmSan/ClarityGuard |

## Active Model

| Property | Value |
|---|---|
| Active version | ClarityGuard v2 |
| Training checkpoint | 750 |
| Base model | Unsloth Gemma 4 E4B IT BNB 4-bit |
| Architecture | Gemma 4 |
| Parameters | 7.52B |
| Quantization | GGUF / Q4_K_M |
| Main file | `ClarityGuard-v2.gguf` |
| Multimodal projector | `mmproj-ClarityGuard-v2.gguf` |
| Model context metadata | 131072 tokens |
| HF Space deployed context | 12288 tokens |
| Inference runtime | llama.cpp / llama-server |

Deprecated checkpoint-375 artifacts are not the active deployment files. The final public deployment uses `ClarityGuard-v2.gguf` and `mmproj-ClarityGuard-v2.gguf`.

## What ClarityGuard Does

ClarityGuard analyzes communication structure rather than judging the user or inferring the sender's intent. It is designed for situations such as:

- ambiguous work instructions
- vague feedback like "be more proactive"
- ghost ownership such as "we need to fix this"
- undefined deadlines like "soon" or "ASAP"
- social or workplace messages where the expected action is implied but not stated
- image-supported or multimodal situations through the Gemma 4 vision projector

The model returns concrete output: structural analysis, cognitive protection, a read-back clarification question, and a follow-up plan when ambiguity persists.

## C.F.R.V.A. Framework

C.F.R.V.A. is the analysis framework created by Carlos Lengemann (2026), published under CC BY 4.0.

Published: [doi.org/10.5281/zenodo.19636473](https://doi.org/10.5281/zenodo.19636473)

| Factor | Detects |
|---|---|
| Context | Undeclared assumptions or missing background |
| Framing | Undefined terms or missing measurable criteria |
| Responsibility | Unclear ownership, ghost "we", or missing actor |
| Validation | Approval implicitly conditioned on not asking questions |
| Ambiguity | Jargon, metaphor, indirect language, or unsupported instructions |

Each dimension is scored from 0 to 10, for a maximum score of 50.

| Score | Response mode |
|---|---|
| 0-10 | Clear message. Confirm briefly. |
| 11-20 | General clarity issue. Name the ambiguity and suggest one confirmation question. |
| 21-30 | Moderate ambiguity. Full analysis plus cognitive protection and clarification. |
| 31-50 | Maximum alert. Full four-step analysis plus follow-up plan. |

## Architecture

```text
User message / image
        |
        v
HF Space Gradio interface
        |
        v
RAG context from ClarityGuard knowledge files
        |
        v
llama-server + ClarityGuard-v2.gguf
        |
        v
C.F.R.V.A. structured analysis
```

Training data was generated with a teacher/student workflow:

```text
Kaggle manipulation-conversation dataset
        |
        v
filter_dataset.py
2,444 coworker examples + ~556 contrast examples
        |
        v
Dify batch pipeline + gemma4:31b-cloud (Ollama) teacher + RAG
        |
        v
2,999 valid C.F.R.V.A. analyses
        |
        v
Unsloth Studio fine-tuning on Gemma 4 E4B IT
        |
        v
ClarityGuard v2 checkpoint 750
        |
        v
GGUF Q4_K_M export for llama.cpp
```

## How Gemma 4 Was Used

### Teacher model

gemma4:31b-cloud (Ollama) was used through a Dify self-hosted pipeline to generate structured C.F.R.V.A. responses from a curated dataset. The teacher pipeline included RAG context from the Chatty 231051 framework and the author's communication-analysis material.

### Student model

Gemma 4 E4B IT was fine-tuned with Unsloth Studio for the final ClarityGuard v2 checkpoint. The active checkpoint is `750`, exported to GGUF and deployed with llama.cpp for local-first, GPU-accelerated inference.

### Why E4B

- large enough to internalize the four-step C.F.R.V.A. response format
- small enough to run on consumer and hosted GPUs
- supports multimodal inference through `mmproj-ClarityGuard-v2.gguf`
- deployable through llama.cpp without relying on a closed inference provider

## Training and Evaluation Artifacts

The training dataset (2,999 C.F.R.V.A. analyses) is published on
[HuggingFace Datasets](https://huggingface.co/datasets/CharlieBonito/clarity-guard-training-data).

Additional scripts and audit artifacts used to build and evaluate the system:

- `data/filter_dataset.py` - selects the 3,000-example training pool
- `dify/dify_batch_chat.py` - sends examples through Dify/Gemma teacher generation
- `dify/build_training_merge_dify_outputs.py` - merges teacher outputs
- `training/train_clarityguard_sft.py` - reproducible QLoRA SFT training script
- `training/train_config.yaml` - versioned training configuration and legacy pilot profiles
- `Documentation/manipulational_conversation_responses_all.txt` - consolidated valid teacher responses
- `Documentation/Kaggle final.xlsx` - comparison/evaluation artifact
- `Documentation/Kaggle Answers.xlsx` - 56 workplace scenario evaluations
- `Documentation/README_RESULTADOS_DATASET.txt` - dataset result summary

Some large source JSONL files are intentionally not committed to keep the repository lightweight. The published model weights and available evaluation artifacts are hosted separately on Hugging Face and in `Documentation/`.

## Reproducibility Notes

This repository contains the complete pipeline used to build ClarityGuard, but some
dependencies require setup that is specific to the author's environment:

### Training Dataset

The complete training dataset (2,999 C.F.R.V.A. analyses) is available on
HuggingFace Datasets:

```
https://huggingface.co/datasets/CharlieBonito/clarity-guard-training-data
```

Load it directly:
```python
from datasets import load_dataset
dataset = load_dataset("CharlieBonito/clarity-guard-training-data", split="train")
```

### Teacher RAG Sources

The RAG context used during teacher generation (Chatty 231051 framework and
organizational gaslighting book extract) is included in `docs/`:
- `docs/rag_chatty_framework.md` — Chatty 231051 framework
- `docs/rag_libro_manipulacion.md` — Book extract on manipulation awareness

These are **proprietary** (all rights reserved by Carlos Lengemann) and are
provided for reference and reproducibility of the teacher pipeline.

### Data Pipeline (for reference)
- The source dataset `manipulational_conversation.jsonl` is a Kaggle dataset
  (not included). The filter script (`data/filter_dataset.py`) documents how to
  produce `curated_dataset_3000.jsonl` from it.
- The Dify pipeline (`dify/dify_batch_chat.py`) assumes a self-hosted Dify instance with
  Ollama running `gemma4:31b-cloud` as the teacher model. You will need to set
  `DIFY_API_KEY` and `DIFY_BASE_URL` environment variables to use it.
- This pipeline is provided for reference; the final dataset is already published.

### Training
- The production model (ClarityGuard v2) was fine-tuned via **Unsloth Studio UI**,
  not via the CLI script. The CLI script `train_clarityguard_sft.py` documents the
  pipeline structure and the `clarityguard_v2` profile in `train_config.yaml` records
  the hyperparameters used.
- To reproduce the training yourself: configure a profile with `model_name: unsloth/gemma-4-e4b-it`
  and provide a compatible JSONL dataset in `messages` format.

### Export
- The final GGUF export was done through the Unsloth Studio UI export function.
  The script `export/exportar.py` provides a reference for HF Safetensors export
  from a LoRA checkpoint, but is not the exact process used for the published GGUF.

### Inference Scripts
- `deploy/run-llama-server-q4.sh` expects `llama-server` in PATH or configured via the
  `LLAMA` environment variable. Override defaults with environment variables.

## Deployment

Download the model files from Hugging Face:

```bash
huggingface-cli download CharlieBonito/clarity-guard-gemma4-7b \
  ClarityGuard-v2.gguf \
  mmproj-ClarityGuard-v2.gguf \
  --local-dir deploy
```

Then run llama.cpp:

```bash
cd deploy
chmod +x run-llama-server-q4.sh
./run-llama-server-q4.sh
```

Default runtime:

```text
model: ClarityGuard-v2.gguf
mmproj: mmproj-ClarityGuard-v2.gguf
host: 0.0.0.0
port: 8081
context: 12288
gpu layers: 999
```

Test the server:

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "They said: \"We need to fix this soon.\" What does that mean?"
      }
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

## Repository Structure

```text
ClarityGuard/
├── README.md
├── LICENSE
├── requirements.txt
├── Kagglecompetition.txt
├── data/
│   ├── filter_dataset.py
│   ├── build_gemma4_e2b_final.py
│   └── build_gemma4_sft_internalize.py
├── dify/
│   ├── dify_batch_chat.py
│   ├── dify_batch_kaggle.py
│   └── build_training_merge_dify_outputs.py
├── training/
│   ├── train_clarityguard_sft.py
│   ├── train_config.yaml
│   ├── prepare_dataset.py
│   └── normalize_assistant_language.py
├── export/
│   └── exportar.py
├── deploy/
│   ├── run-llama-server-q4.sh
│   ├── run-llama-server.sh
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── modal_deploy.py
│   └── clarity-e4b-vision.service
├── prompts/
│   ├── clarityguard_prompt_v4.4.txt
│   └── clarityguard_prompt_v4.7.txt
├── docs/
│   ├── manifiesto_cfrva_v1.0.md
│   ├── rag_chatty_framework.md        ← RAG source: Chatty 231051 framework
│   └── rag_libro_manipulacion.md      ← RAG source: book extract (gaslighting org.)
└── Documentation/
    ├── README_RESULTADOS_DATASET.txt
    ├── manipulational_conversation_responses_all.txt
    ├── Kaggle final.xlsx
    ├── Kaggle Answers.xlsx
    └── respuestas/
```

## Hackathon Tracks

ClarityGuard is positioned for:

- Digital Equity & Inclusivity
- Safety & Trust
- Unsloth Special Track
- llama.cpp Special Track

## License

- Code: Apache 2.0
- C.F.R.V.A. framework: CC BY 4.0, attribution to Carlos Lengemann (2026), DOI: https://doi.org/10.5281/zenodo.19636473
- Proprietary RAG/source writing: all rights reserved by Carlos Lengemann unless otherwise stated

## Author

Carlos Lengemann

Model weights: https://huggingface.co/CharlieBonito/clarity-guard-gemma4-7b

Live demo: https://huggingface.co/spaces/CharlieBonito/ClarityGuardAgent

## Kaggle Submission Notes

The canonical Kaggle writeup is available in [`KAGGLE_WRITEUP_CORRECTED.md`](KAGGLE_WRITEUP_CORRECTED.md). It uses the current ClarityGuard v2 wording:

- active files: `ClarityGuard-v2.gguf` and `mmproj-ClarityGuard-v2.gguf`
- checkpoint 375 artifacts are legacy exports
- training sequence length: 4096 tokens
- inference context depends on runtime deployment
- per-device batch size: 1
- gradient accumulation: 4
- effective batch size: 4
- LoRA r/alpha is not published as definitive because the final exact value is not independently verified

The Unsloth screenshots are summarized in [`Documentation/UNSLOTH_SCREENSHOT_FINDINGS.md`](Documentation/UNSLOTH_SCREENSHOT_FINDINGS.md). They should be used as reproducibility evidence, not as the complete original training log.
