#!/usr/bin/env python3
"""
Filas JSONL 5000–5999 (1-based, inclusive).
Salida: manipulational_conversation_responses5000.txt
Estado: dify_batch_ranges5000.json → row_ranges["5000-5999"].next_row_1based
  (reanudación automática: no cambies --from-row/--to-row).
Espera 30 s entre peticiones.

Checkpoint: última Pregunta en el .txt = 5092 (respuesta OK).
  Estado JSON: next_row_1based=5093; pendiente 5093–5999.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_MAIN = _DIR / "dify_batch_chat.py"

if __name__ == "__main__":
    sys.argv = [
        str(_MAIN),
        "--from-row",
        "5000",
        "--to-row",
        "5999",
        "--output",
        str(_DIR / "manipulational_conversation_responses5000.txt"),
        "--ranges-file",
        str(_DIR / "dify_batch_ranges5000.json"),
        "--delay",
        "30",
    ]
    runpy.run_path(str(_MAIN), run_name="__main__")
