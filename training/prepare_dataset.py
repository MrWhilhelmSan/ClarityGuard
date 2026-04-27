"""
Preparación de ejemplos chat para Gemma + Unsloth SFT.

- Opcional: quita la línea [dataset_id=...] del primer user (inferencia más limpia).
- Convierte rol assistant → model (plantilla Gemma).
- Opcional: antepone system_prompt al primer turno user.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

DATASET_ID_LINE = re.compile(r"\n*\[dataset_id=[^\]]+\]\s*\n*", re.MULTILINE)


def strip_dataset_id_from_text(text: str) -> str:
    return DATASET_ID_LINE.sub("\n\n", text, count=1).strip()


def normalize_roles_for_gemma(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            role = "model"
        out.append({"role": role, "content": m["content"]})
    return out


def inject_system_into_first_user(
    messages: list[dict[str, str]], system_prompt: str
) -> list[dict[str, str]]:
    if not system_prompt.strip():
        return messages
    msgs = copy.deepcopy(messages)
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            msgs[i] = {
                "role": "user",
                "content": system_prompt.strip() + "\n\n" + m["content"],
            }
            break
    return msgs


def prepare_messages_row(
    row: dict[str, Any],
    *,
    strip_dataset_id_line: bool,
    system_prompt: str | None,
) -> dict[str, Any]:
    raw_messages = row["messages"]
    messages = normalize_roles_for_gemma(raw_messages)
    if strip_dataset_id_line:
        fixed: list[dict[str, str]] = []
        first_user_done = False
        for m in messages:
            c = m["content"]
            if m["role"] == "user" and not first_user_done:
                c = strip_dataset_id_from_text(c)
                first_user_done = True
            fixed.append({"role": m["role"], "content": c})
        messages = fixed
    if system_prompt:
        messages = inject_system_into_first_user(messages, system_prompt)
    meta = {k: v for k, v in row.items() if k != "messages"}
    return {"messages": messages, **meta}


def load_jsonl_messages(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids: list[str] = []
    bad = 0
    for row in rows:
        did = row.get("dataset_id")
        if did is not None:
            ids.append(str(did))
        msgs = row.get("messages")
        if not msgs or not isinstance(msgs, list):
            bad += 1
            continue
        roles = [m.get("role") for m in msgs]
        if roles != ["user", "assistant"]:
            # Permitir ya normalizado user/model
            if roles != ["user", "model"]:
                bad += 1
    dup = len(ids) - len(set(ids)) if ids else 0
    return {
        "num_rows": len(rows),
        "num_dataset_ids": len(ids),
        "duplicate_dataset_ids": dup,
        "bad_rows": bad,
    }


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
