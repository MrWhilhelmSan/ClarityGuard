#!/usr/bin/env python3
"""
Unifica respuestas de dify_batch_chat1, 2000, 3000, 4000 y 5000 en un solo JSONL
en clean/, con el mismo esquema que gemma4_unsloth_sft.jsonl (metadatos JSONL +
model_* + conversations/messages/text Gemma 4).

Rangos (como en los scripts *chat*.py):
  chat1   → filas 1–1000     → manipulational_conversation_responses1.txt
  chat2000→ 1001–1999        → ..._responses2000.txt
  chat3000→ 2001–2999        → ..._responses3000.txt
  chat4000→ 2000–2999        → ..._responses4000.txt
  chat5000→ 5000–5999        → ..._responses5000.txt

Solape 2001–2999 entre 3000 y 4000: se conserva la primera respuesta válida
según el orden 3000 → 4000 (la 4000 completa la fila 2000, que 3000 no cubre).
"""
from __future__ import annotations

import json
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CLEAN = _DIR / "clean"
_OUT_NAME = "gemma4_unsloth_sft_chat1_5000.jsonl"

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_merge", str(_DIR / "build_training_merge_dify_outputs.py")
)
_merge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_merge)

load_jsonl = _merge.load_jsonl
collect_valid_rows = _merge.collect_valid_rows


def to_gemma4_text_no_bos(user_content: str, assistant_content: str) -> str:
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

    # Orden: 1, 2000, 3000, 4000, 5000. Primera respuesta válida por fila gana.
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
        (
            _DIR / "manipulational_conversation_responses4000.txt",
            "manipulational_conversation_responses4000.txt",
            2000,
            2999,
        ),
        (
            _DIR / "manipulational_conversation_responses5000.txt",
            "manipulational_conversation_responses5000.txt",
            5000,
            5999,
        ),
    ]

    by_row: dict[int, tuple[str, str, str]] = {}
    for pth, label, rmin, rmax in batches:
        if not pth.is_file():
            continue
        for row_n, q, a, src in collect_valid_rows(pth, label, rmin, rmax):
            if row_n not in by_row:
                by_row[row_n] = (q, a, src)

    merged: list[dict] = []
    for row_n in sorted(by_row):
        if row_n < 1 or row_n > total:
            continue
        q, a, src = by_row[row_n]
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

    print(f"Ejemplos (filas únicas): {len(merged)} → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
