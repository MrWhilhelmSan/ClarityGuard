#!/usr/bin/env python3
"""
Filas JSONL 2001–2999 (1-based, inclusive).
Salida: manipulational_conversation_responses3000.txt
Estado: dify_batch_ranges3000.json → next_row_1based en row_ranges["2001-2999"].
Reanuda sola desde la última fila guardada; --from-row/--to-row fijan solo el rango permitido.
Espera 30 s entre peticiones (igual que los demás scripts *000).

Checkpoint: última Pregunta en el .txt = 2109 con [ERROR API] (SSE/timeout).
  Estado JSON alineado: next_row_1based=2109 (reintento de esa fila), pendiente 2109–2999.
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
        "2001",
        "--to-row",
        "2999",
        "--output",
        str(_DIR / "manipulational_conversation_responses3000.txt"),
        "--ranges-file",
        str(_DIR / "dify_batch_ranges3000.json"),
        "--delay",
        "30",
    ]
    runpy.run_path(str(_MAIN), run_name="__main__")
