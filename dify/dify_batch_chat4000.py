#!/usr/bin/env python3
"""
Filas JSONL 3000–4000 (1-based, inclusive).
Empieza justo después de dify_batch_chat3000.py (2001–2999), sin dejar hueco en la 3000.
Salida: manipulational_conversation_responses4000.txt
Estado: dify_batch_ranges4000.json → row_ranges["3000-4000"].next_row_1based
Espera 30 s entre peticiones (igual que los demás scripts *000).
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
        "3000",
        "--to-row",
        "4000",
        "--output",
        str(_DIR / "manipulational_conversation_responses4000.txt"),
        "--ranges-file",
        str(_DIR / "dify_batch_ranges4000.json"),
        "--delay",
        "30",
    ]
    runpy.run_path(str(_MAIN), run_name="__main__")
