import os
import sys
from pathlib import Path

# No hace falta `activate` un venv: lo que importa es QUÉ `python` ejecuta PowerShell
# (mira la línea siguiente). Si ves rutas bajo `.unsloth\studio\unsloth_studio\`, estás
# usando el Python empaquetado de Unsloth Studio (es un entorno aislado aunque no lo actives).
print(f"--- Intérprete activo: {sys.executable} ---", flush=True)

import unsloth  # aplicar parches antes de cargar el resto de transformers
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# Gemma 4 exige transformers reciente; Unsloth Studio a veces arrastra 4.57.x sin gemma4.
# Pip del entorno Studio: ...\unsloth_studio\Scripts\pip.exe install "transformers==5.5.0"
# (<=5.5.0 cumple unsloth; 5.5.3 supera el tope <=5.5.0 y pip avisará de conflicto).
#
# Export GGUF solo desde la *UI* de Studio (no aplica a este .py): timeouts al cargar
# modelos grandes — p. ej. _wait_response("loaded", timeout=300) en
# studio/backend/core/export/orchestrator.py; upstream ha subido el límite (~900s) en commits recientes.
if "gemma4" not in CONFIG_MAPPING:
    print(
        "Este intérprete no reconoce model_type 'gemma4'.\n"
        f"  Python: {sys.executable}\n"
        f"  transformers: {transformers.__version__}\n"
        "Instala en el MISMO entorno con el que ejecutas este script, por ejemplo:\n"
        f'  "{os.path.join(os.environ.get("USERPROFILE", ""), ".unsloth", "studio", "unsloth_studio", "Scripts", "pip.exe")}" install "transformers==5.5.0"',
        file=sys.stderr,
    )
    sys.exit(1)

from unsloth import FastVisionModel

_REPO_ROOT = Path(__file__).resolve().parent

# 1. Configuración de rutas (checkpoint LoRA; sobrescribe con UNSLOTH_CHECKPOINT_DIR si quieres otro)
_run_id = "unsloth_gemma-4-e2b-it-unsloth-bnb-4bit_1775848152"
_checkpoint = "checkpoint-576"
checkpoint_path = os.environ.get(
    "UNSLOTH_CHECKPOINT_DIR",
    str(
        Path(os.environ.get("USERPROFILE", ""))
        / ".unsloth"
        / "studio"
        / "outputs"
        / _run_id
        / _checkpoint
    ),
)
# Salida HF junto al proyecto (listo para ollama_desde_hf.py). Sobrescribe con UNSLOTH_HF_EXPORT_DIR.
destination_path = os.environ.get(
    "UNSLOTH_HF_EXPORT_DIR",
    str(_REPO_ROOT / "dist" / "hf_merged"),
)

# Gemma-4-E2B es Gemma4ForConditionalGeneration (multimodal). Aunque solo entrenes texto,
# el checkpoint sigue siendo esa arquitectura: FastVisionModel / FastModel es la vía correcta.
load_in_4bit = True
# Si la base está en NF4/FP4, Unsloth avisa que merged_16bit no aplica y aborta la fusión
# (devuelve None sin error). Para fusionar LoRA sobre la base 4bit usa forced_merged_4bit.
# Para un merge real en float16 necesitas base 16bit/BF16 (más VRAM) o flujo distinto (p. ej. GGUF).
save_method = os.environ.get(
    "UNSLOTH_EXPORT_SAVE_METHOD",
    "forced_merged_4bit" if load_in_4bit else "merged_16bit",
)

print("--- Cargando Gemma 4 (E2B) y adaptador LoRA ---")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = checkpoint_path,
    load_in_4bit = load_in_4bit,
)

Path(destination_path).mkdir(parents=True, exist_ok=True)

print(f"--- Exportando a {destination_path} con save_method={save_method!r} ---")

if save_method == "forced_merged_4bit":
    # Transformers 5.5 + bitsandbytes: revert_weight_conversion usa reverse_op de
    # Bnb4bitDeserialize, que lanza NotImplementedError. Unsloth llama save_pretrained()
    # sin argumentos; aquí guardamos con save_original_format=False para omitir esa ruta.
    print("--- Fusionando LoRA (merge_and_unload) y guardando (save_original_format=False) ---")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        destination_path,
        save_original_format = False,
        safe_serialization = True,
    )
    tokenizer.save_pretrained(destination_path)
else:
    model.save_pretrained_merged(
        destination_path,
        tokenizer,
        save_method = save_method,
    )

_dest = Path(destination_path)
_st = list(_dest.glob("*.safetensors"))
_bin = list(_dest.glob("*.bin"))
_bytes = sum(f.stat().st_size for f in _st + _bin)
print("--- Exportacion HF completada con exito. ---")
print(
    f"--- Resumen: {_dest} | {len(_st)} x safetensors, {len(_bin)} x .bin, ~{_bytes / (1024**3):.2f} GiB ---"
)
if os.environ.get("UNSLOTH_EXPORT_VERBOSE"):
    print(
        "(CUDA 12 en el banner vs cu130 en Torch: suelen ser lecturas distintas; es normal.)"
    )