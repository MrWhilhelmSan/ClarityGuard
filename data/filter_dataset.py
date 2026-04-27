#!/usr/bin/env python3
"""
Filtra manipulational_conversation.jsonl para producir un dataset curado de 3,000 registros:
  - 2,444 registros con context_type == "coworker" (todos)
  - ~200 registros "neutral" (sin manipulacion) de otros contextos, los mas largos
  - ~356 registros manipulativos de otros contextos (friend/romantic/family), los mas largos

Uso:
  python3 filter_dataset.py
  python3 filter_dataset.py --input mi_archivo.jsonl --output mi_salida.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

TARGET_TOTAL = 3000
NEUTRAL_COUNT = 200
COWORKER_CONTEXT = "coworker"


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Linea {line_no}: JSON invalido: {e}", file=sys.stderr)
                sys.exit(1)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_stats(records: list[dict]) -> None:
    print(f"\n{'='*60}")
    print(f"  Dataset curado: {len(records)} registros")
    print(f"{'='*60}")

    ctx = Counter(r.get("context_type", "?") for r in records)
    print("\n  Por contexto:")
    for k, v in ctx.most_common():
        print(f"    {k}: {v}")

    mt = Counter(r.get("manipulation_type", "?") for r in records)
    print("\n  Por tipo de manipulacion:")
    for k, v in mt.most_common():
        print(f"    {k}: {v}")

    manip = sum(1 for r in records if r.get("is_manipulation"))
    print(f"\n  Manipulativos: {manip}  |  Neutrales: {len(records) - manip}")

    wc = [r.get("word_count_total", 0) for r in records]
    if wc:
        wc_sorted = sorted(wc)
        print(f"\n  Conteo de palabras:")
        print(f"    Min: {wc_sorted[0]}  Max: {wc_sorted[-1]}  "
              f"Media: {sum(wc)/len(wc):.1f}  Mediana: {wc_sorted[len(wc)//2]}")

    print()


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "manipulational_conversation.jsonl"
    default_output = script_dir / "curated_dataset_3000.jsonl"

    p = argparse.ArgumentParser(description="Filtra JSONL a dataset curado de 3,000 registros.")
    p.add_argument("--input", type=Path, default=default_input, help="JSONL de entrada")
    p.add_argument("--output", type=Path, default=default_output, help="JSONL de salida")
    args = p.parse_args()

    if not args.input.is_file():
        print(f"No existe: {args.input}", file=sys.stderr)
        return 1

    print(f"Leyendo {args.input} ...")
    records = load_jsonl(args.input)
    print(f"  Total: {len(records)} registros")

    coworker = [r for r in records if r.get("context_type") == COWORKER_CONTEXT]
    others = [r for r in records if r.get("context_type") != COWORKER_CONTEXT]

    print(f"  Coworker: {len(coworker)}")
    print(f"  Otros contextos: {len(others)}")

    remaining = TARGET_TOTAL - len(coworker)
    manipulative_count = remaining - NEUTRAL_COUNT

    neutrals_pool = [r for r in others if r.get("manipulation_type") == "neutral"]
    neutrals_pool.sort(key=lambda r: r.get("word_count_total", 0), reverse=True)
    neutrals = neutrals_pool[:NEUTRAL_COUNT]
    neutral_ids = {r.get("conversation_id") for r in neutrals}

    manipulative_pool = [
        r for r in others
        if r.get("is_manipulation") and r.get("conversation_id") not in neutral_ids
    ]
    manipulative_pool.sort(key=lambda r: r.get("word_count_total", 0), reverse=True)
    manipulative_others = manipulative_pool[:manipulative_count]

    final = coworker + neutrals + manipulative_others

    print(f"\nSeleccion:")
    print(f"  Coworker:              {len(coworker)}")
    print(f"  Neutral (contraste):   {len(neutrals)}")
    print(f"  Manipulativos otros:   {len(manipulative_others)}")
    print(f"  TOTAL:                 {len(final)}")

    if len(final) != TARGET_TOTAL:
        print(f"\n  AVISO: el total es {len(final)}, no {TARGET_TOTAL}.", file=sys.stderr)

    write_jsonl(args.output, final)
    print(f"\nEscrito: {args.output}")

    print_stats(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
