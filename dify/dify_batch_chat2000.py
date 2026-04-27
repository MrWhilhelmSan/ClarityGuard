#!/usr/bin/env python3
"""
Filas JSONL 1001–1999 (1-based, inclusive).
Salida: manipulational_conversation_responses2000.txt
Estado: dify_batch_ranges2000.json → next_row_1based en row_ranges["1001-1999"].
Reanuda sola desde la última fila guardada; --from-row/--to-row fijan solo el rango permitido.
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
        "1001",
        "--to-row",
        "1999",
        "--output",
        str(_DIR / "manipulational_conversation_responses2000.txt"),
        "--ranges-file",
        str(_DIR / "dify_batch_ranges2000.json"),
        "--delay",
        "30",
    ]
    runpy.run_path(str(_MAIN), run_name="__main__")
