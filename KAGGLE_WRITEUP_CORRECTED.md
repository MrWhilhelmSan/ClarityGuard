# ClarityGuard — Neuro-Inclusive Communication Analysis with Gemma 4

**Author:** Carlos Lengemann — Psychologist & AI Engineer, Manizales, Colombia  
**Tracks:** Digital Equity & Inclusivity · Safety & Trust · Unsloth Special Track · llama.cpp Special Track

## The Problem

Ambiguity in workplace communication is a daily obstacle for neurodivergent people — autistic adults in particular. When a manager writes "we need to fix that soon," most neurotypical colleagues fill in the blanks using social shortcuts: pattern matching, confirmation bias, conformity. They guess, and often guess correctly enough to avoid conflict.

Autistic people do not use those shortcuts — not because of a deficit, but because they process communication literally and logically. When a message lacks a defined subject, a measurable outcome, a concrete deadline, or an explicit owner, the correct logical response is: *this message cannot be executed with certainty*. The confusion is valid. The message is incomplete.

The problem is not cognitive. It is structural: the message was transmitted without the minimum parameters required for execution. Yet in most workplaces, asking for clarification is socially penalized. ClarityGuard was built to reduce that forced choice: execute an ambiguous instruction and risk failure, or ask and risk a social penalty.

## The Psychological Foundation

As a clinical psychologist, I noticed that many workplace communication failures reported by neurodivergent people share structural characteristics with manipulation tactics — not because the sender is always malicious, but because vague and manipulative communication often use the same structure: undefined terms, missing accountability, conditioned validation, and implicit penalties for asking questions.

This observation led to the **C.F.R.V.A. Framework** (Lengemann, 2026), an original analytical model published under CC BY 4.0 with DOI [10.5281/zenodo.19636473](https://doi.org/10.5281/zenodo.19636473). It scores messages across five structural dimensions:

| Dimension | What It Detects |
|-----------|-----------------|
| **C** — Undeclared Context | Implicit assumptions never stated explicitly |
| **F** — Diffuse Focusing | No measurable criteria for what "done" looks like |
| **R** — Covert Redirection | Priority shift without signaling |
| **V** — Conditioned Validation | Approval depends on *not* asking questions |
| **A** — Linguistic Ambiguity | Jargon, metaphors, undefined terms |

Each dimension is scored 0-10 for a maximum of 50. A score above 30 triggers the full 4-step protocol. Step 2, cognitive protection, reframes the problem from cognitive deficit to communication bug: confusion is the correct response to an incomplete message.

## The Solution

ClarityGuard is a fine-tuned Gemma 4 E4B model that analyzes the **structure** of messages, not the user's ability to understand them. It applies C.F.R.V.A. and produces:

**Step 1 — Analysis:** Structural breakdown of what the message contains vs. what is missing.  
**Step 2 — Cognitive Protection:** Logical validation that confusion is structurally justified.  
**Step 3 — Read-Back:** A concrete clarification script the user can send verbatim.  
**Step 4 — Follow-Up Plan:** Binary Choice Decomposition if the reply remains abstract.

The system also includes 3-mode input triage: casual questions get conversational responses, minor phrasing confusions get brief clarifications, and real communication conflicts trigger the full C.F.R.V.A. protocol.

## How Gemma 4 Was Used

### Teacher: gemma4:31b-cloud via Ollama

The 31B cloud model served as teacher to generate the training dataset. A public Kaggle dataset of 10,000 manipulation-related workplace conversations was curated to 3,000 examples: 2,444 coworker-domain conversations plus 556 contrast examples from other relationship contexts. Each was processed through a self-hosted Dify pipeline with RAG context from the Chatty 231051 symbolic framework and the author's manipulation-awareness writing. Using the ClarityGuard v4.7 system prompt, the teacher produced 2,999 valid structured C.F.R.V.A. analyses.

An additional 56 real-world workplace scenarios commonly reported by autistic and neurodivergent people were processed through the same pipeline and included as a benchmark evaluation set.

### Student: Gemma 4 E4B IT fine-tuned with Unsloth

Gemma 4 E4B IT was chosen for native multimodal support, edge deployability, and enough capacity to learn the 4-step C.F.R.V.A. output format. An initial E2B attempt was insufficient for consistent structured output.

Fine-tuning used **Unsloth QLoRA 4-bit** on a local Linux/KachiOS machine with an RTX 5070 Ti and 16 GB VRAM. The per-device batch size was kept at 1 with 4 gradient accumulation steps, for an effective batch size of 4, to avoid VRAM spikes under the 16 GB ceiling.

| Hyperparameter | Value |
|----------------|-------|
| Adapter method | QLoRA 4-bit via Unsloth Studio |
| LoRA r / alpha | Not published as definitive; final value not independently verified |
| Load in 4-bit | True |
| Max sequence length | 4,096 |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Optimizer | AdamW 8-bit |
| Precision | bf16 |

Available Unsloth screenshots document a replicated run of the workflow and confirm Gemma 4 E4B, QLoRA 4-bit, 4096-token sequence length, bf16, AdamW 8-bit, batch 1 x 4 accumulation, and RTX 5070 Ti VRAM pressure near the 16 GB ceiling. They are reproducibility evidence, not the complete original training log.

The active production model is **ClarityGuard v2**, exported to GGUF Q4_K_M. The production files are `ClarityGuard-v2.gguf` and `mmproj-ClarityGuard-v2.gguf`. Older `Checkpoint-375-*` files are legacy exports and are not the current submission weights.

## Architecture

```text
User Message / Image
        |
        v
Gradio Interface (HuggingFace Space)
        |
        +-- Jina Embeddings v3 RAG
        |   +-- Chatty 231051 + manipulation-awareness source material
        |
        v
ClarityGuard v2 — Gemma 4 E4B (7.52B params)
GGUF Q4_K_M · llama-server + CUDA 12.6
        |
        v
C.F.R.V.A. score (/50) + 4-step structured response
```

The HuggingFace Space uses a precompiled llama-server binary targeting CUDA architectures 75 and 89 for T4/L4 compatibility. An initial local CUDA 13 binary failed on HuggingFace with `libcudart.so.13: cannot open shared object file`; the fix was recompiling inside a CUDA 12.6 devel container and shipping the binary via Git LFS.

## Evaluation

The 56-case benchmark compares four conditions across real autistic-reported workplace scenarios:

| Condition | Description |
|-----------|-------------|
| Base model, no RAG | Gemma 4 E4B with generic prompt |
| Fine-tuned, no RAG | ClarityGuard v2 with generic prompt |
| Base model + RAG | Gemma 4 E4B with RAG context |
| Fine-tuned + RAG | ClarityGuard v2 with full RAG |

Fine-tuning produced structural consistency in the C.F.R.V.A. output format that neither RAG alone nor the base model achieved. The base model tends to give general advice; the fine-tuned model consistently applies the 4-step protocol, scores the message, and produces a concrete read-back script.

## Technical Challenges Overcome

**E2B insufficient capacity:** Gemma 4 E2B could not reliably learn the structured 4-step output format. Switching to E4B resolved this.

**CUDA version mismatch on HuggingFace:** llama.cpp had to be rebuilt in a CUDA 12.6 devel container with L4/T4-compatible architectures.

**Vision support vs. Ollama:** Exporting to Ollama did not preserve the full multimodal path, so deployment uses llama.cpp directly with the mmproj file.

**VRAM management during training:** With the RTX 5070 Ti at 16 GB, training used per-device batch size 1 with gradient accumulation 4, keeping the effective batch size at 4 while avoiding OOM spikes.

**Teacher output quality:** Early teacher batches included malformed responses. A filtering and validation pipeline produced the final 2,999 clean samples.

## Why This Matters

Autistic adults face severe employment barriers in many studies, often due to communication friction rather than inability to do the work. Tools that make structural ambiguity visible and provide a concrete, socially neutral clarification script directly reduce a barrier to economic participation.

ClarityGuard also connects neurodivergent communication support to AI reliability: systems that resist vague terms, social code, and unsupported assumptions are more useful for everyone.

## Resources

| Resource | Link |
|----------|------|
| Video | YouTube link |
| Live Demo | https://huggingface.co/spaces/CharlieBonito/ClarityGuardAgent |
| Model Weights | https://huggingface.co/CharlieBonito/clarity-guard-gemma4-7b |
| Code Repository | https://github.com/MrWhilhelmSan/ClarityGuard |
| 56-Case Benchmark | `Documentation/Kaggle_Comparacion_56.xlsx` |
| C.F.R.V.A. Framework | https://doi.org/10.5281/zenodo.19636473 |

*C.F.R.V.A. Framework — Carlos Lengemann (2026) · CC BY 4.0 · DOI: [10.5281/zenodo.19636473](https://doi.org/10.5281/zenodo.19636473)*  
*Code — Apache 2.0*  
*Built in Manizales, Colombia for the neurodivergent community worldwide.*
