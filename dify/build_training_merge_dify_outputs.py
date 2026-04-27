#!/usr/bin/env python3
"""
Arma el dataset solo con respuestas de los tres batches Dify (sin mezclar otros orígenes):

  - chat1 → manipulational_conversation_responses1.txt, filas JSONL 1–1000
  - chat2000 → manipulational_conversation_responses2000.txt, filas 1001–1999
  - chat3000 → manipulational_conversation_responses3000.txt, filas 2001–2999

Solo entran respuestas válidas (no [ERROR API] ni stream vacío).

Genera:
  - manipulational_conversation_with_model_responses_train_ready.jsonl
  - manipulational_conversation_with_model_responses_train_ready_unsloth_format.jsonl
    (mismos metadatos que train_ready + id/instruction/output/text/messages chat)
  - nuevo_training_gemma4_2b.jsonl (igual que unsloth_format)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_DIR = Path(__file__).resolve().parent

QUERY_PREFIX_EN = (
    "Please reply entirely in English.\n\n"
    "They sent me the following — what's going on?\n\n"
)


def format_conversation(record: dict) -> str:
    parts = []
    for m in record.get("messages", []):
        sp = m.get("speaker", "?")
        tx = (m.get("text") or "").strip()
        parts.append(f"{sp}: {tx}")
    return "\n".join(parts)


def build_query(record: dict) -> str:
    meta = (
        f"[dataset_id={record.get('conversation_id', '')} | "
        f"type={record.get('manipulation_type', '')} | "
        f"context={record.get('context_type', '')} | "
        f"is_manipulation={record.get('is_manipulation', '')}]\n\n"
    )
    return QUERY_PREFIX_EN + meta + format_conversation(record)


def is_error_response(answer: str) -> bool:
    a = (answer or "").strip()
    return a.startswith("[ERROR API]") or a.startswith("[sin texto en el stream")


def parse_dify_txt(path: Path, source_label: str) -> dict[int, tuple[str, str, str]]:
    """Devuelve mapa fila_1based -> (query, answer, source_file) desde un .txt de batch."""
    text = path.read_text(encoding="utf-8")
    sep = "=" * 72
    out: dict[int, tuple[str, str, str]] = {}

    pattern = re.compile(
        rf"(?:^|\n){re.escape(sep)}\nPregunta (\d+)\n{re.escape(sep)}\n"
        r"(.*?)\n\nRespuesta \1\n"
        rf"{re.escape(sep)}\n"
        r"(.*?)(?=\n\n" + re.escape(sep) + r"\nPregunta |\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        n = int(m.group(1))
        q = m.group(2).strip()
        a = m.group(3).strip()
        if m_cid := re.search(r"\n\[dify_conversation_id:[^\]]+\]\s*$", a):
            a = a[: m_cid.start()].strip()
        out[n] = (q, a, source_label)
    return out


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_unsloth_row(seq_id: int, rec: dict) -> dict:
    """Incluye todos los campos del JSONL original + model_* + campos Unsloth."""
    q = rec["model_query"]
    a = rec["model_response"]
    text = (
        "<start_of_turn>user\n"
        f"{q}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{a}\n"
        "<end_of_turn>"
    )
    row = dict(rec)
    # `messages` pasa a ser el chat user/assistant; conservar el diálogo A/B del dataset.
    if "messages" in rec:
        row["dataset_messages"] = rec["messages"]
    row["id"] = seq_id
    row["messages"] = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]
    row["instruction"] = q
    row["input"] = ""
    row["output"] = a
    row["text"] = text
    return row


def collect_valid_rows(
    path: Path,
    source_label: str,
    row_min: int,
    row_max: int,
) -> list[tuple[int, str, str, str]]:
    """Lista de (fila_1based, query, answer, source) dentro del rango, solo respuestas válidas."""
    if not path.is_file():
        return []
    parsed = parse_dify_txt(path, source_label)
    out: list[tuple[int, str, str, str]] = []
    for row_n in sorted(parsed):
        if row_n < row_min or row_n > row_max:
            continue
        q, a, src = parsed[row_n]
        if is_error_response(a) or not a.strip():
            continue
        out.append((row_n, q, a, src))
    return out


def main() -> int:
    jsonl_path = _DIR / "manipulational_conversation.jsonl"
    records = load_jsonl(jsonl_path)
    total = len(records)

    # Solo los tres archivos de batch; rangos alineados con dify_batch_chat{1,2000,3000}.py
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
    counts: list[tuple[str, int]] = []
    for pth, label, rmin, rmax in batches:
        rows = collect_valid_rows(pth, label, rmin, rmax)
        picked.extend(rows)
        counts.append((label, len(rows)))

    picked.sort(key=lambda t: t[0])

    merged: list[dict] = []
    for row_n, q, a, src in picked:
        if row_n < 1 or row_n > total:
            continue
        base = dict(records[row_n - 1])
        out = dict(base)
        out["model_query"] = q
        out["model_response"] = a
        out["model_response_status"] = "ok"
        out["model_response_source_file"] = src
        merged.append(out)

    out_ready = _DIR / "manipulational_conversation_with_model_responses_train_ready.jsonl"
    out_unsloth = _DIR / "manipulational_conversation_with_model_responses_train_ready_unsloth_format.jsonl"
    out_nuevo = _DIR / "nuevo_training_gemma4_2b.jsonl"

    with out_ready.open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with out_unsloth.open("w", encoding="utf-8") as f:
        for j, rec in enumerate(merged, start=1):
            f.write(json.dumps(to_unsloth_row(j, rec), ensure_ascii=False) + "\n")

    out_nuevo.write_text(out_unsloth.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Total ejemplos (solo 3 batches, sin fallback): {len(merged)}")
    for label, c in counts:
        print(f"  {label}: {c}")
    print(f"Escrito: {out_ready.name}, {out_unsloth.name}, {out_nuevo.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
