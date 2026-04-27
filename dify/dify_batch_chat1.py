#!/usr/bin/env python3
"""
Filas JSONL 1–1000 (1-based, inclusive).
Salida: manipulational_conversation_responses1.txt
Estado: dify_batch_ranges1.json → la fila siguiente es next_row_1based en row_ranges["1-1000"]
(reanudación automática; no cambies --from-row/--to-row salvo que quieras otro rango).
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
        "1",
        "--to-row",
        "1000",
        "--output",
        str(_DIR / "manipulational_conversation_responses1.txt"),
        "--ranges-file",
        str(_DIR / "dify_batch_ranges1.json"),
        "--delay",
        "30",
    ]
    runpy.run_path(str(_MAIN), run_name="__main__")
