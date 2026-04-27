#!/usr/bin/env python3
"""
Lee Questions Kaggle.xlsx, envía cada pregunta a la API de Dify y guarda las respuestas.

Uso:
  python3 dify_batch_kaggle.py                    # Procesa desde la última fila guardada
  python3 dify_batch_kaggle.py --redo             # Reinicia desde la fila 1
  python3 dify_batch_kaggle.py --from-row 10      # Inicia desde fila 10
  python3 dify_batch_kaggle.py --delay 30         # 30 segundos entre peticiones

Salida: kaggle_responses.txt
Estado: dify_batch_kaggle_state.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# API de Dify
DIFY_API_KEY_PLACEHOLDER = "app-zoIlZf0Xq8ID5mb1v0AKBEfQ"
DIFY_BASE_URL_DEFAULT = "http://localhost/v1"

# Prefijo enviado al modelo
QUERY_PREFIX = (
    "Please reply entirely in English.\n\n"
    "Analyze the following workplace situation and provide insights:\n\n"
)


def read_excel_column(xlsx_path: Path) -> list[str]:
    """
    Lee un archivo Excel .xlsx y extrae el contenido de la primera columna.
    Retorna una lista de strings (una por fila).
    No requiere dependencias externas - usa zipfile + xml.
    """
    rows: list[str] = []

    with zipfile.ZipFile(xlsx_path, 'r') as z:
        # Leer shared strings
        shared_strings: list[str] = []
        with z.open('xl/sharedStrings.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            for si in root.findall('.//main:si', ns):
                text_parts = []
                for t in si.findall('.//main:t', ns):
                    if t.text:
                        text_parts.append(t.text)
                shared_strings.append(''.join(text_parts))

        # Leer hoja 1
        with z.open('xl/worksheets/sheet1.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

            for row in root.findall('.//main:row', ns):
                cell = row.find('main:c', ns)
                if cell is not None:
                    cell_type = cell.get('t')
                    value_elem = cell.find('main:v', ns)
                    if value_elem is not None and value_elem.text:
                        if cell_type == 's':
                            # Referencia a shared string
                            idx = int(value_elem.text)
                            if 0 <= idx < len(shared_strings):
                                rows.append(shared_strings[idx])
                            else:
                                rows.append('')
                        else:
                            # Valor directo
                            rows.append(value_elem.text)
                    else:
                        rows.append('')
                else:
                    rows.append('')

    return rows


def _parse_sse_line_objects(line: str) -> list[dict]:
    """Extrae objetos JSON de una línea SSE."""
    line = line.strip()
    if not line:
        return []
    dec = json.JSONDecoder()
    objs: list[dict] = []
    i = 0
    n = len(line)
    while i < n:
        while i < n and line[i] in " \t":
            i += 1
        if i >= n:
            break
        if line.startswith("data:", i):
            j = i + 5
            while j < n and line[j] in " \t":
                j += 1
            if j < n and line[j : j + 7] == "[DONE]":
                i = j + 7
                continue
            try:
                val, end = dec.raw_decode(line, j)
                if isinstance(val, dict):
                    objs.append(val)
                i = end
            except json.JSONDecodeError:
                i = j + 1
        else:
            i += 1
    return objs


def _consume_chat_sse(resp) -> tuple[str, str | None]:
    """Lee text/event-stream de Dify y concatena los deltas de `answer`."""
    answer_parts: list[str] = []
    conversation_id: str | None = None
    buf = ""

    def handle_obj(obj: dict) -> None:
        nonlocal conversation_id
        ev = obj.get("event")
        if ev == "ping":
            return
        if ev == "error":
            raise RuntimeError(f"SSE error: {obj.get('message', obj)}")
        cid = obj.get("conversation_id")
        if cid:
            conversation_id = cid
        if obj.get("answer") is None:
            return
        if ev in (None, "message", "agent_message", "text_chunk"):
            answer_parts.append(str(obj["answer"]))
            return
        if ev == "message_end":
            return
        if isinstance(obj.get("answer"), str):
            answer_parts.append(str(obj["answer"]))

    while True:
        chunk = resp.read(16384)
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.rstrip("\r")
            for obj in _parse_sse_line_objects(line):
                handle_obj(obj)

    for obj in _parse_sse_line_objects(buf):
        handle_obj(obj)

    full = "".join(answer_parts)
    return full, conversation_id


def post_chat_message(
    base_url: str,
    api_key: str,
    query: str,
    user_id: str,
    timeout: int,
) -> tuple[str, str | None]:
    """Envía mensaje a Dify y retorna (respuesta, conversation_id)."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        url = f"{root}/chat-messages"
    else:
        url = f"{root}/v1/chat-messages"

    payload = {
        "inputs": {},
        "query": query,
        "response_mode": "streaming",
        "user": user_id,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            full, conv_id = _consume_chat_sse(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e

    if not full.strip():
        return "[sin texto en el stream SSE]", conv_id
    return full, conv_id


def append_block(
    out_path: Path,
    index: int,
    query: str,
    answer: str,
    conv_id: str | None,
    flush: bool = True,
) -> None:
    """Agrega un bloque pregunta/respuesta al archivo de salida."""
    sep = "=" * 72
    block = (
        f"\n{sep}\n"
        f"Pregunta {index}\n"
        f"{sep}\n"
        f"{query}\n\n"
        f"Respuesta {index}\n"
        f"{sep}\n"
        f"{answer}\n"
    )
    if conv_id:
        block += f"\n[dify_conversation_id: {conv_id}]\n"
    block += "\n"

    with out_path.open("a", encoding="utf-8") as f:
        f.write(block)
        if flush:
            f.flush()
            os.fsync(f.fileno())


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_xlsx = script_dir / "Questions Kaggle.xlsx"
    default_out = script_dir / "kaggle_responses.txt"
    default_state = script_dir / "dify_batch_kaggle_state.json"

    p = argparse.ArgumentParser(description="Procesa Excel y envía preguntas a Dify.")
    p.add_argument("--xlsx", type=Path, default=default_xlsx, help="Ruta al archivo Excel")
    p.add_argument("--output", type=Path, default=default_out, help="Archivo de salida")
    p.add_argument("--state-file", type=Path, default=default_state, help="Archivo de estado JSON")
    p.add_argument(
        "--base-url",
        default=os.environ.get("DIFY_BASE_URL", DIFY_BASE_URL_DEFAULT),
        help="URL base de la API de Dify",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("DIFY_API_KEY", DIFY_API_KEY_PLACEHOLDER),
        help="API key de Dify",
    )
    p.add_argument("--delay", type=int, default=30, help="Segundos entre peticiones")
    p.add_argument("--timeout", type=int, default=600, help="Timeout HTTP (s)")
    p.add_argument("--user-id", default="kaggle-batch", help="Campo user para Dify")
    p.add_argument("--from-row", type=int, default=None, help="Fila inicial (1-based)")
    p.add_argument("--limit", type=int, default=None, help="Número máximo de filas a procesar")
    p.add_argument("--redo", action="store_true", help="Reiniciar desde la fila 1")

    args = p.parse_args()

    if not args.api_key:
        print("Falta API key.", file=sys.stderr)
        return 1

    if not args.xlsx.is_file():
        print(f"No existe el archivo Excel: {args.xlsx}", file=sys.stderr)
        return 1

    # Leer Excel
    print(f"Leyendo: {args.xlsx}")
    questions = read_excel_column(args.xlsx)
    total = len(questions)
    print(f"Total de filas: {total}")

    if total == 0:
        print("El archivo Excel está vacío.", file=sys.stderr)
        return 1

    # Cargar estado
    state = load_state(args.state_file)

    # Determinar fila inicial
    if args.redo:
        start_row = 1
        state["next_row"] = 1
        state["completed"] = False
        save_state(args.state_file, state)
    elif args.from_row is not None:
        start_row = args.from_row
        state["next_row"] = start_row
        state["completed"] = False
        save_state(args.state_file, state)
    else:
        start_row = state.get("next_row", 1)

    if start_row > total:
        print(f"Todas las {total} filas ya fueron procesadas.", file=sys.stderr)
        return 0

    # Límite de filas a procesar
    end_row = total
    if args.limit is not None:
        end_row = min(start_row - 1 + args.limit, total)

    # Limpiar archivo de salida si es nuevo proceso
    if start_row == 1 and args.output.exists():
        args.output.unlink()

    print(f"\nProcesando filas {start_row} a {end_row} (de {total}). delay={args.delay}s\n")
    print(f"Salida: {args.output}")
    print(f"API: {args.base_url.rstrip('/')}/chat-messages\n")

    # Procesar cada fila
    for i in range(start_row - 1, end_row):
        idx_1based = i + 1
        question = questions[i]

        if not question or not question.strip():
            print(f"[{idx_1based}/{total}] Fila vacía, saltando...")
            state["next_row"] = idx_1based + 1
            save_state(args.state_file, state)
            continue

        query = QUERY_PREFIX + question.strip()

        print(f"[{idx_1based}/{total}] Enviando...", flush=True)
        try:
            answer, conv_id = post_chat_message(
                args.base_url,
                args.api_key,
                query,
                args.user_id,
                args.timeout,
            )
        except Exception as e:
            err = f"[ERROR API] {e}"
            print(err, file=sys.stderr)
            append_block(args.output, idx_1based, query, err, None)
            state["next_row"] = idx_1based
            state["completed"] = False
            save_state(args.state_file, state)
            return 2

        append_block(args.output, idx_1based, query, answer, conv_id)
        state["next_row"] = idx_1based + 1
        if state["next_row"] > end_row:
            state["completed"] = True
        save_state(args.state_file, state)
        print(f"[{idx_1based}/{end_row}] Guardado. conv_id={conv_id!r}", flush=True)

        if i < end_row - 1:
            print(f"Esperando {args.delay}s...", flush=True)
            time.sleep(args.delay)

    print(f"\nProceso completado. Filas {start_row} a {end_row} procesadas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
