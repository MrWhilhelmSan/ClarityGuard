"""
Unifica encabezados frecuentes en español dentro del rol assistant → inglés.

Uso:
  python normalize_assistant_language.py --input ../manipulational_conversation_responses_train_messages.jsonl \\
      --output ../manipulational_conversation_responses_train_messages_normalized.jsonl

Idempotente para la mayoría de patrones (no reemplaza si ya está en inglés).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Orden: cadenas largas primero para evitar sustituciones parciales incorrectas.
REPLACEMENTS: list[tuple[str, str]] = [
    ("**[ClarityGuard] Detectado:**", "**[ClarityGuard] Detected:**"),
    ("**ANÁLISIS DE LOS 3 JUECES:**", "**ANALYSIS OF THE 3 JUDGES:**"),
    ("**JUEZ 1 - COHERENCIA COMUNICATIVA:**", "**JUDGE 1 - COMMUNICATION COHERENCE:**"),
    ("**JUEZ 2 - ARQUITECTURA DE CLARIDAD (C.C.R.V.A.):**", "**JUDGE 2 - CLARITY ARCHITECTURE (C.C.R.V.A.):**"),
    ("**JUEZ 3 - COHERENCIA FACTUAL Y OPERATIVA:**", "**JUDGE 3 - FACTUAL AND OPERATIONAL COHERENCE:**"),
    ("**PUNTAJE TOTAL:**", "**TOTAL SCORE:**"),
    ("**DEFENSA ASERTIVA DIRECTA:**", "**DIRECT ASSERTIVE DEFENSE:**"),
    ("**SUGERENCIA DE COMUNICACIÓN ASERTIVA:**", "**SUGGESTED ASSERTIVE COMMUNICATION:**"),
    ("**PATRÓN A MANEJAR - RESISTENCIA ESPERADA:**", "**PATTERN TO MANAGE - EXPECTED RESISTANCE:**"),
    ("**PLAN DE SEGUIMIENTO ITERATIVO:**", "**ITERATIVE FOLLOW-UP PLAN:**"),
    ("**PRINCIPIO DE APRENDIZAJE:**", "**LEARNING PRINCIPLE:**"),
    ("**OBJETIVO DE ESTA ITERACIÓN:**", "**OBJECTIVE OF THIS ITERATION:**"),
    ("**Nota importante:**", "**Important note:**"),
    ("ALTA PROBABILIDAD DE COMUNICACIÓN MANIPULATIVA (GASLIGHTING)", "HIGH PROBABILITY OF MANIPULATIVE COMMUNICATION (GASLIGHTING)"),
    ("ALTA PROBABILIDAD DE COMUNICACIÓN MANIPULATIVA", "HIGH PROBABILITY OF MANIPULATIVE COMMUNICATION"),
    ("ALTA PROBABILIDAD DE COMUNICACIÓN POCO CLARA CON INDUCCIÓN DE CULPA", "HIGH PROBABILITY OF UNCLEAR COMMUNICATION WITH GUILT INDUCTION"),
    ("Patrón de **", "Pattern of **"),
    ("Invalidación directa**", "Direct invalidation**"),
]


def normalize_assistant_text(text: str) -> str:
    s = text
    for old, new in REPLACEMENTS:
        s = s.replace(old, new)
    return s


def process_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    n = 0
    changed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages")
            if not msgs:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                continue
            new_msgs = []
            row_changed = False
            for m in msgs:
                if m.get("role") == "assistant":
                    before = m["content"]
                    after = normalize_assistant_text(before)
                    if after != before:
                        row_changed = True
                    new_msgs.append({"role": "assistant", "content": after})
                else:
                    new_msgs.append(m)
            row["messages"] = new_msgs
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if row_changed:
                changed += 1
    return n, changed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    total, changed = process_file(args.input, args.output)
    print(f"Filas escritas: {total}, filas con cambios en assistant: {changed}")


if __name__ == "__main__":
    main()
