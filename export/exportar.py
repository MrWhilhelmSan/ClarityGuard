#!/usr/bin/env python3
"""
Exportar checkpoint LoRA de Unsloth a formato HuggingFace Safetensors.

Uso:
  export UNSLOTH_CHECKPOINT_DIR=/ruta/al/checkpoint-750
  python exportar.py

Tambien soportado via variables de entorno:
  UNSLOTH_CHECKPOINT_DIR  — ruta al checkpoint de Unsloth
  UNSLOTH_HF_EXPORT_DIR   — directorio de salida (default: ./dist/hf_merged)
  UNSLOTH_EXPORT_SAVE_METHOD — "forced_merged_4bit" | "merged_16bit" (default: auto)

NOTA: Este script se proporciona como referencia del pipeline de exportacion.
      El modelo publico final (ClarityGuard v2) fue exportado a GGUF Q4_K_M
      via Unsloth Studio UI, no con este script. Los GGUF estan disponibles en:
      https://huggingface.co/CharlieBonito/clarity-guard-gemma4-7b
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import unsloth  # noqa: E402
import transformers  # noqa: E402
from transformers.models.auto.configuration_auto import CONFIG_MAPPING  # noqa: E402

print(f"--- Intérprete: {sys.executable} ---", flush=True)

if "gemma4" not in CONFIG_MAPPING:
    print(
        "ERROR: transformers no reconoce model_type 'gemma4'.\n"
        f"  transformers: {transformers.__version__}\n"
        "Instala transformers >= 5.5.0 en el mismo entorno.",
        file=sys.stderr,
    )
    sys.exit(1)

from unsloth import FastVisionModel  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent

# Checkpoint LoRA — configurar via variable de entorno
checkpoint_path = os.environ.get(
    "UNSLOTH_CHECKPOINT_DIR",
    str(_REPO_ROOT / "checkpoint-750"),
)

# Directorio de salida
destination_path = os.environ.get(
    "UNSLOTH_HF_EXPORT_DIR",
    str(_REPO_ROOT / "dist" / "hf_merged"),
)

load_in_4bit = True
save_method = os.environ.get(
    "UNSLOTH_EXPORT_SAVE_METHOD",
    "forced_merged_4bit" if load_in_4bit else "merged_16bit",
)

print(f"--- Cargando checkpoint: {checkpoint_path} ---")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=checkpoint_path,
    load_in_4bit=load_in_4bit,
)

Path(destination_path).mkdir(parents=True, exist_ok=True)

print(f"--- Exportando a {destination_path} con save_method={save_method!r} ---")

if save_method == "forced_merged_4bit":
    print("--- Fusionando LoRA (merge_and_unload) y guardando ---")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        destination_path,
        save_original_format=False,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(destination_path)
else:
    model.save_pretrained_merged(
        destination_path,
        tokenizer,
        save_method=save_method,
    )

_dest = Path(destination_path)
_st = list(_dest.glob("*.safetensors"))
_bin = list(_dest.glob("*.bin"))
_bytes = sum(f.stat().st_size for f in _st + _bin)
print("--- Exportacion completada ---")
print(
    f"--- {_dest} | {len(_st)} x safetensors, {len(_bin)} x .bin, ~{_bytes / (1024**3):.2f} GiB ---"
)
