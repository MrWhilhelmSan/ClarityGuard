#!/usr/bin/env python3
"""
Genera un único JSONL en clean/ para SFT de Gemma 4 con Unsloth:
  - Preguntas = filas de manipulational_conversation.jsonl
  - Respuestas = manipulational_conversation_responses{1,2000,3000}.txt (solo válidas)

Formato alineado con la guía Gemma 4 de Unsloth (train_on_responses_only usa
  instruction_part = "<|turn>user\\n", response_part = "<|turn>model\\n"):
  https://unsloth.ai/docs/models/gemma-4

El campo `text` omite <bos> (el processor lo añade al entrenar).
"""
from __future__ import annotations

import json
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CLEAN = _DIR / "clean"
_OUT_NAME = "gemma4_unsloth_sft.jsonl"

# Reutilizar parse/validación del merge existente
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_merge", str(_DIR / "build_training_merge_dify_outputs.py")
)
_merge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_merge)

load_jsonl = _merge.load_jsonl
collect_valid_rows = _merge.collect_valid_rows
is_error_response = _merge.is_error_response


def to_gemma4_text_no_bos(user_content: str, assistant_content: str) -> str:
    """Plantilla conversación Gemma 4 sin token inicial (Unsloth añade <bos>)."""
    u = (user_content or "").strip()
    a = (assistant_content or "").strip()
    return (
        "<|turn>user\n"
        f"{u}\n"
        "<turn|>\n"
        "<|turn>model\n"
        f"{a}\n"
        "<turn|>\n"
    )


def main() -> int:
    jsonl_path = _DIR / "manipulational_conversation.jsonl"
    records = load_jsonl(jsonl_path)
    total = len(records)

    batches = [
        (
            _DIR / "manipulational_conversation_responses1.txt",
            "manipulational_conversation_responses1.txt",
            1,
            1000,
        ),
        (
            _DIR / "manipulational_conversation_responses2000.txt",
            "manipulational_conversation_responses2000.txt",
            1001,
            1999,
        ),
        (
            _DIR / "manipulational_conversation_responses3000.txt",
            "manipulational_conversation_responses3000.txt",
            2001,
            2999,
        ),
    ]

    picked: list[tuple[int, str, str, str]] = []
    for pth, label, rmin, rmax in batches:
        picked.extend(collect_valid_rows(pth, label, rmin, rmax))
    picked.sort(key=lambda t: t[0])

    merged: list[dict] = []
    for row_n, q, a, src in picked:
        if row_n < 1 or row_n > total:
            continue
        base = dict(records[row_n - 1])
        out = dict(base)
        out["jsonl_row_1based"] = row_n
        out["model_query"] = q
        out["model_response"] = a
        out["model_response_status"] = "ok"
        out["model_response_source_file"] = src
        out["conversations"] = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        out["messages"] = list(out["conversations"])
        out["text"] = to_gemma4_text_no_bos(q, a)
        merged.append(out)

    _CLEAN.mkdir(parents=True, exist_ok=True)
    out_path = _CLEAN / _OUT_NAME
    with out_path.open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Ejemplos escritos: {len(merged)} → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
