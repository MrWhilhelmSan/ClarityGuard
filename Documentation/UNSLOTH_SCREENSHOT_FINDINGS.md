# Unsloth Screenshot Findings

These screenshots appear to document a replicated Unsloth Studio run, not necessarily the original full ClarityGuard v2 training run. Use them as reproducibility evidence unless confirmed otherwise.

## Confirmed From Screenshots

| Item | Observed value |
|------|----------------|
| Base model | `unsloth/gemma-4-e4b-it` |
| Method | QLoRA 4-bit |
| Max sequence length | 4096 |
| Precision | bf16 enabled, fp16 disabled |
| Optimizer | AdamW 8-bit |
| Weight decay | 0.001 |
| Seed | 3407 |
| Dataset size | 2999 samples |
| Dataset processes | 5 |
| Packing | enabled |
| Assistant completions only | enabled |
| Warmup steps | 5 |
| Save steps | 30 |
| Save strategy | steps |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| VRAM observed during training | about 14.33 / 15.92 GB |
| GPU utilization observed | 100% |

## Replicated Run Values

The terminal screenshot shows this exact training configuration:

| Item | Observed value |
|------|----------------|
| Max steps | 30 |
| Epochs | 1 |
| Per-device train batch size | 1 |
| Gradient accumulation steps | 4 |
| Total batch size | 4 |
| Learning rate | 0.0002 |
| LR scheduler | cosine |
| Logging steps | 1 |
| Trainable parameters | 21,200,896 / 8,017,357,344 |
| Trainable percentage | 0.26% |

## LoRA Settings Visible In UI

The visible UI screenshots show:

| Item | Observed value |
|------|----------------|
| Rank | 8 |
| Alpha | 8 |
| Dropout | 0.00 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Memory option | LoftQ selected |

One screenshot also shows an earlier/alternate panel with batch size 2, gradient accumulation 4, and linear scheduler. A later screenshot plus the terminal output show batch size 1, gradient accumulation 4, and cosine scheduler. Treat the terminal output as the stronger evidence for the replicated run.

## Loss / Checkpoint Evidence

The screenshots show:

- Start of training: step 1 / 30 with loss around 8.7593.
- Terminal logs include early losses around 8.759, 8.916, and 8.9.
- Export screen shows a run with 26 checkpoints.
- Export screen lists checkpoint losses including:
  - run head / selected item: loss 1.1104
  - checkpoint-120: loss 1.0077
  - checkpoint-150: loss 0.9834
  - checkpoint-180: loss 0.9971
  - checkpoint-210: loss 0.9331
  - checkpoint-240: loss 0.9264
  - checkpoint-270: loss 0.8511
  - checkpoint-30: loss 1.6726

The export screenshot does not show checkpoint 750. It supports that there were multiple checkpoints in a longer run, but it does not by itself prove the active v2 checkpoint 750.

## Interpretation

These images are useful for explaining the replicated workflow:

- local Unsloth Studio on RTX 5070 Ti
- Gemma 4 E4B base
- QLoRA 4-bit
- 4096-token training sequence length
- bf16
- AdamW 8-bit
- micro-batch 1 with 4 gradient accumulation steps
- high VRAM usage close to the 16 GB ceiling

They should not replace the canonical ClarityGuard v2 production statement unless the original training logs confirm the same LoRA rank/alpha, learning rate, scheduler, and checkpoint/loss metrics.
