#!/usr/bin/env python3
"""
Envía a Dify las preguntas de un spreadsheet XLSX y guarda las respuestas
en un XLSX nuevo, una respuesta por celda en la columna A.

Por defecto lee `Kaggle Answers.xlsx` en esta misma carpeta. Si el archivo
tiene encabezados `Qestion/Answer` o `Question/Answer`, toma las preguntas de
la columna A debajo del encabezado. Si no, toma las celdas no vacías de la
fila 1. Cada pregunta se envía a Dify exactamente como aparece en el XLSX.

La salida por defecto es el primer nombre libre: `Kaggle 1.xlsx`,
`Kaggle 2.xlsx`, `Kaggle 3.xlsx`, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

# Busca y reemplaza este valor por tu API key real de Dify (única ocurrencia de la clave en el repo).
DIFY_API_KEY_PLACEHOLDER = "app-kiy8fXvj8IbioaoIeWMi68UP"

DIFY_BASE_URL_DEFAULT = "http://localhost/v1"

XML_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XML_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def resolve_default_jsonl(script_dir: Path) -> Path:
    """Prioridad: misma carpeta que el script (dify_batch), patrón manipulational_conversation*.jsonl."""
    home = Path.home()
    exact = script_dir / "manipulational_conversation.jsonl"
    if exact.is_file():
        return exact
    matches = sorted(script_dir.glob("manipulational_conversation*.jsonl"))
    if matches:
        return matches[0]
    fallback = [
        home / "Downloads" / "manipulational_conversation.jsonl",
        Path("/mnt/c/Users/carlo/Downloads/manipulational_conversation.jsonl"),
    ]
    for p in fallback:
        if p.is_file():
            return p
    return exact


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
    body = meta + format_conversation(record)
    return QUERY_PREFIX_EN + body


def _parse_sse_line_objects(line: str) -> list[dict]:
    """Extrae objetos JSON de una línea SSE (puede haber varios `data: {...}` seguidos)."""
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
    """
    Lee text/event-stream de Dify y concatena los deltas de `answer`.
    Necesario para Agent Chat App (no soporta blocking).
    """
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
        # Deltas del modelo (chat y agent suelen usar `message`)
        if ev in (None, "message", "agent_message", "text_chunk"):
            answer_parts.append(str(obj["answer"]))
            return
        if ev == "message_end":
            return
        # Otros eventos con texto (por si la versión de Dify cambia)
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
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        url = f"{root}/chat-messages"
    else:
        url = f"{root}/v1/chat-messages"

    # Agent Chat App solo admite streaming (SSE), no blocking.
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
        return "[sin texto en el stream SSE; revisa eventos del agente en Dify]", conv_id
    return full, conv_id


def append_block(
    out_path: Path,
    index: int,
    query: str,
    answer: str,
    conv_id: str | None,
    flush: bool = True,
) -> None:
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


def chunk_row_bounds(chunk_index_0: int, chunk_size: int, total: int) -> tuple[int, int]:
    """Índices globales 1-based [start, end] inclusive."""
    start_1 = chunk_index_0 * chunk_size + 1
    end_1 = min((chunk_index_0 + 1) * chunk_size, total)
    return start_1, end_1


def load_ranges_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_ranges_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def get_chunk_state(state: dict, chunk_key: str) -> dict:
    chunks = state.setdefault("chunks", {})
    return chunks.setdefault(chunk_key, {})


def chunk_label(
    chunk_index_0: int,
    chunk_size: int,
    total: int,
    state: dict,
) -> str:
    """Texto de estado para el menú."""
    start_1, end_1 = chunk_row_bounds(chunk_index_0, chunk_size, total)
    key = str(chunk_index_0)
    cs = get_chunk_state(state, key)
    next_row = cs.get("next_row_1based")
    completed = cs.get("completed", False)

    if completed or (next_row is not None and next_row > end_1):
        return f"✓ YA PROCESADO (completo) — filas {start_1}–{end_1}"
    if next_row is None or next_row <= start_1:
        return f"○ pendiente — filas {start_1}–{end_1}"
    return f"~ en curso — siguiente fila {next_row} (hasta {end_1})"


def is_chunk_fully_done(
    chunk_index_0: int,
    chunk_size: int,
    total: int,
    state: dict,
) -> bool:
    _, end_1 = chunk_row_bounds(chunk_index_0, chunk_size, total)
    key = str(chunk_index_0)
    cs = state.get("chunks", {}).get(key, {})
    if cs.get("completed"):
        return True
    nr = cs.get("next_row_1based")
    return nr is not None and nr > end_1


def pick_chunk_interactive(
    num_chunks: int,
    chunk_size: int,
    total: int,
    state: dict,
    redo: bool,
) -> int | None:
    """Devuelve chunk_index_0 o None si sale."""
    print(f"\nTotal filas: {total}. Tamaño de cada bloque: {chunk_size}.\n")
    print("Bloques (elige un número). Estado = si ya trabajaste ese rango antes:\n")
    for c in range(num_chunks):
        start_1, end_1 = chunk_row_bounds(c, chunk_size, total)
        lbl = chunk_label(c, chunk_size, total, state)
        print(f"  [{c + 1}] Filas {start_1} – {end_1}  →  {lbl}")
    print()

    while True:
        raw = input(f"Selecciona bloque (1–{num_chunks}), o 'q' para salir: ").strip().lower()
        if raw in ("q", "quit", "salir"):
            return None
        try:
            choice = int(raw)
        except ValueError:
            print("Introduce un número válido.")
            continue
        if choice < 1 or choice > num_chunks:
            print(f"Debe estar entre 1 y {num_chunks}.")
            continue
        chunk_0 = choice - 1
        if is_chunk_fully_done(chunk_0, chunk_size, total, state) and not redo:
            print(
                "\n>>> Ese rango YA fue procesado por completo. "
                "Elige otro bloque o ejecuta con --redo para forzar de nuevo.\n"
            )
            continue
        return chunk_0


def _row_range_state_key(start_1: int, end_1: int) -> str:
    return f"{start_1}-{end_1}"


def get_row_range_state(state: dict, start_1: int, end_1: int) -> dict:
    ranges = state.setdefault("row_ranges", {})
    key = _row_range_state_key(start_1, end_1)
    return ranges.setdefault(key, {})


def is_row_range_fully_done(
    start_1: int,
    end_1: int,
    total: int,
    state: dict,
) -> bool:
    if start_1 > total:
        return True
    rs = get_row_range_state(state, start_1, end_1)
    if rs.get("completed"):
        return True
    nr = rs.get("next_row_1based")
    eff_end = min(end_1, total)
    return nr is not None and nr > eff_end


def run_row_range(
    *,
    start_1: int,
    end_1: int,
    records: list[dict],
    state: dict,
    ranges_path: Path,
    args: argparse.Namespace,
    redo: bool,
) -> int:
    """Procesa filas 1-based [start_1, end_1] inclusive (recortado a total del JSONL)."""
    total = len(records)
    if start_1 < 1 or end_1 < start_1:
        print("Rango inválido: --from-row y --to-row (1-based, from <= to).", file=sys.stderr)
        return 1
    if start_1 > total:
        print(f"--from-row {start_1} está fuera del JSONL (total {total}).", file=sys.stderr)
        return 1

    eff_end = min(end_1, total)
    rs = get_row_range_state(state, start_1, end_1)

    if is_row_range_fully_done(start_1, end_1, total, state) and not redo:
        print(
            f"Rango {start_1}–{eff_end} ya está procesado por completo. Usa --redo para repetir.",
            file=sys.stderr,
        )
        return 1

    if redo:
        rs["next_row_1based"] = start_1
        rs["completed"] = False
        save_ranges_state(ranges_path, state)

    next_row = rs.get("next_row_1based")
    if next_row is None or next_row < start_1:
        next_row = start_1
    if next_row > eff_end:
        print("Nada que hacer en este rango.", file=sys.stderr)
        return 0

    i_start = next_row - 1
    i_end_exclusive = eff_end

    print(
        f"\nProcesando rango {start_1}–{eff_end} (de {total} filas). delay={args.delay}s\n",
        flush=True,
    )
    print(f"Salida: {args.output}")
    print(f"API: {args.base_url.rstrip('/')}/chat-messages\n")

    for i in range(i_start, i_end_exclusive):
        idx_1based = i + 1
        record = records[i]
        query = build_query(record)

        print(f"[rango {start_1}–{eff_end}] [{idx_1based}/{total}] Enviando (nuevo chat)...", flush=True)
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
            rs["next_row_1based"] = idx_1based
            rs["completed"] = False
            save_ranges_state(ranges_path, state)
            return 2

        append_block(args.output, idx_1based, query, answer, conv_id)
        rs["next_row_1based"] = idx_1based + 1
        if rs["next_row_1based"] > eff_end:
            rs["completed"] = True
        else:
            rs["completed"] = False
        save_ranges_state(ranges_path, state)
        print(f"[{idx_1based}/{total}] Guardado. conv_id={conv_id!r}", flush=True)

        if i < i_end_exclusive - 1:
            print(f"Esperando {args.delay}s...", flush=True)
            time.sleep(args.delay)

    print(f"\nRango {start_1}–{eff_end} terminado.")
    return 0


def run_chunk(
    *,
    chunk_index_0: int,
    chunk_size: int,
    records: list[dict],
    state: dict,
    ranges_path: Path,
    args: argparse.Namespace,
    redo: bool,
) -> int:
    total = len(records)
    start_1, end_1 = chunk_row_bounds(chunk_index_0, chunk_size, total)
    key = str(chunk_index_0)
    cs = get_chunk_state(state, key)

    if is_chunk_fully_done(chunk_index_0, chunk_size, total, state) and not redo:
        print(f"Bloque {chunk_index_0 + 1} (filas {start_1}–{end_1}) ya está completo. Usa --redo para repetir.", file=sys.stderr)
        return 1

    if redo:
        cs["next_row_1based"] = start_1
        cs["completed"] = False
        save_ranges_state(ranges_path, state)

    next_row = cs.get("next_row_1based")
    if next_row is None or next_row < start_1:
        next_row = start_1
    if next_row > end_1:
        print("Nada que hacer en este bloque.", file=sys.stderr)
        return 0

    # Índices 0-based en lista
    i_start = next_row - 1
    i_end_exclusive = end_1  # range es hasta end_1 inclusive → i_end_exclusive en range: end_1

    print(f"\nProcesando bloque {chunk_index_0 + 1}: filas {next_row}–{end_1} (de {total}). delay={args.delay}s\n")
    print(f"Salida: {args.output}")
    print(f"API: {args.base_url.rstrip('/')}/chat-messages\n")

    state["chunk_size"] = chunk_size

    for i in range(i_start, i_end_exclusive):
        idx_1based = i + 1
        record = records[i]
        query = build_query(record)

        print(f"[bloque {chunk_index_0 + 1}] [{idx_1based}/{total}] Enviando (nuevo chat)...", flush=True)
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
            cs["next_row_1based"] = idx_1based
            cs["completed"] = False
            save_ranges_state(ranges_path, state)
            return 2

        append_block(args.output, idx_1based, query, answer, conv_id)
        cs["next_row_1based"] = idx_1based + 1
        if cs["next_row_1based"] > end_1:
            cs["completed"] = True
        else:
            cs["completed"] = False
        save_ranges_state(ranges_path, state)
        print(f"[{idx_1based}/{total}] Guardado. conv_id={conv_id!r}", flush=True)

        if i < i_end_exclusive - 1:
            print(f"Esperando {args.delay}s...", flush=True)
            time.sleep(args.delay)

    print(f"\nBloque {chunk_index_0 + 1} (filas {start_1}–{end_1}) terminado.")
    return 0


def _xlsx_ns(tag: str) -> str:
    return f"{{{XML_NS_MAIN}}}{tag}"


def _cell_column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref.upper())
    if not match:
        return 0
    col = 0
    for ch in match.group(1):
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return col


def _column_name(index_1based: int) -> str:
    name = ""
    n = index_1based
    while n:
        n, rem = divmod(n - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    root = ET.fromstring(data)
    strings: list[str] = []
    for si in root.findall(_xlsx_ns("si")):
        text_parts: list[str] = []
        direct_t = si.find(_xlsx_ns("t"))
        if direct_t is not None and direct_t.text:
            text_parts.append(direct_t.text)
        for t in si.findall(f".//{_xlsx_ns('t')}"):
            if t is direct_t:
                continue
            if t.text:
                text_parts.append(t.text)
        strings.append("".join(text_parts))
    return strings


def _read_workbook_first_sheet_path(zf: zipfile.ZipFile) -> str:
    try:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    except KeyError:
        return "xl/worksheets/sheet1.xml"

    first_sheet = workbook.find(f"{_xlsx_ns('sheets')}/{_xlsx_ns('sheet')}")
    if first_sheet is None:
        return "xl/worksheets/sheet1.xml"

    rel_id = first_sheet.attrib.get(f"{{{XML_NS_REL}}}id")
    if not rel_id:
        return "xl/worksheets/sheet1.xml"

    for rel in rels.findall(f"{{{XML_NS_PKG_REL}}}Relationship"):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib.get("Target", "worksheets/sheet1.xml")
            if target.startswith("/"):
                return target.lstrip("/")
            return f"xl/{target}"
    return "xl/worksheets/sheet1.xml"


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(f".//{_xlsx_ns('t')}"))

    value = cell.find(_xlsx_ns("v"))
    if value is None or value.text is None:
        return ""

    raw = value.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError):
            return ""
    if cell_type == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return raw


def read_first_row_questions(xlsx_path: Path) -> list[str]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet_path = _read_workbook_first_sheet_path(zf)
        sheet = ET.fromstring(zf.read(sheet_path))

    row = sheet.find(f"{_xlsx_ns('sheetData')}/{_xlsx_ns('row')}[@r='1']")
    if row is None:
        row = sheet.find(f"{_xlsx_ns('sheetData')}/{_xlsx_ns('row')}")
    if row is None:
        return []

    cells: list[tuple[int, str]] = []
    for cell in row.findall(_xlsx_ns("c")):
        ref = cell.attrib.get("r", "")
        value = _cell_value(cell, shared_strings)
        if value.strip():
            cells.append((_cell_column_index(ref), value))

    cells.sort(key=lambda item: item[0])
    return [value for _, value in cells]


def _read_sheet_rows(xlsx_path: Path) -> list[list[str]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet_path = _read_workbook_first_sheet_path(zf)
        sheet = ET.fromstring(zf.read(sheet_path))

    rows: list[list[str]] = []
    for row in sheet.findall(f"{_xlsx_ns('sheetData')}/{_xlsx_ns('row')}"):
        values_by_col: dict[int, str] = {}
        max_col = 0
        for cell in row.findall(_xlsx_ns("c")):
            col = _cell_column_index(cell.attrib.get("r", ""))
            if col < 1:
                continue
            values_by_col[col] = _cell_value(cell, shared_strings)
            max_col = max(max_col, col)
        rows.append([values_by_col.get(col, "") for col in range(1, max_col + 1)])
    return rows


def read_kaggle_questions(xlsx_path: Path) -> tuple[list[str], str]:
    rows = _read_sheet_rows(xlsx_path)
    if not rows:
        return [], "spreadsheet vacio"

    first_row = rows[0]
    first_cell = first_row[0].strip().lower() if first_row else ""
    second_cell = first_row[1].strip().lower() if len(first_row) > 1 else ""
    looks_like_question_answer_table = (
        first_cell in {"question", "qestion", "questions", "pregunta", "preguntas"}
        and second_cell in {"answer", "answers", "respuesta", "respuestas"}
    )

    if looks_like_question_answer_table:
        questions = [row[0] for row in rows[1:] if row and row[0].strip()]
        return questions, "columna A debajo del encabezado"

    questions = [value for value in first_row if value.strip()]
    return questions, "fila 1"


def next_kaggle_output_path(directory: Path) -> Path:
    index = 1
    while True:
        candidate = directory / f"Kaggle {index}.xlsx"
        if not candidate.exists():
            return candidate
        index += 1


def write_answers_xlsx(out_path: Path, answers: list[str]) -> None:
    row_xml: list[str] = []
    for row_index, answer in enumerate(answers, start=1):
        cell_ref = f"A{row_index}"
        safe_answer = escape(answer, {'"': "&quot;"})
        row_xml.append(
            f'<row r="{row_index}">'
            f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">{safe_answer}</t></is></c>'
            "</row>"
        )

    last_row = max(1, len(answers))
    sheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{XML_NS_MAIN}" xmlns:r="{XML_NS_REL}">
  <dimension ref="A1:A{last_row}"/>
  <sheetViews><sheetView workbookViewId="0"/></sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <sheetData>
    {''.join(row_xml)}
  </sheetData>
</worksheet>'''

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>''',
        )
        zf.writestr(
            "_rels/.rels",
            f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{XML_NS_PKG_REL}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>''',
        )
        zf.writestr(
            "xl/workbook.xml",
            f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="{XML_NS_MAIN}" xmlns:r="{XML_NS_REL}">
  <sheets><sheet name="Answers" sheetId="1" r:id="rId1"/></sheets>
</workbook>''',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{XML_NS_PKG_REL}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>''',
        )
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr(
            "docProps/app.xml",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>Python</Application>
</Properties>''',
        )
        zf.writestr(
            "docProps/core.xml",
            '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:creator>dify_batch_chat.py</dc:creator>
</cp:coreProperties>''',
        )


def run_kaggle_xlsx(args: argparse.Namespace) -> int:
    if not args.api_key:
        print("Falta API key.", file=sys.stderr)
        return 1
    if not args.input.is_file():
        print(f"No existe el spreadsheet: {args.input}", file=sys.stderr)
        return 1

    questions, source_label = read_kaggle_questions(args.input)
    if not questions:
        print(f"No encontré preguntas en {args.input}", file=sys.stderr)
        return 1

    output_path = args.output if args.output is not None else next_kaggle_output_path(args.input.parent)
    answers: list[str] = []

    print(f"Spreadsheet: {args.input}")
    print(f"Preguntas detectadas ({source_label}): {len(questions)}")
    print(f"Salida: {output_path}")
    print(f"API: {args.base_url.rstrip('/')}/chat-messages\n")

    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] Enviando pregunta...", flush=True)
        conv_id = None
        try:
            answer, conv_id = post_chat_message(
                args.base_url,
                args.api_key,
                question,
                args.user_id,
                args.timeout,
            )
        except Exception as e:
            answer = f"[ERROR API] {e}"
            print(answer, file=sys.stderr)

        answers.append(answer)
        write_answers_xlsx(output_path, answers)
        print(f"[{index}/{len(questions)}] Guardado en columna A. conv_id={conv_id!r}", flush=True)

        if index < len(questions):
            print(f"Esperando {args.delay}s...", flush=True)
            time.sleep(args.delay)

    print(f"\nTerminado. Respuestas guardadas en: {output_path}")
    return 0


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "Kaggle Answers.xlsx"

    p = argparse.ArgumentParser(description="Envía a Dify las preguntas de la fila 1 de un XLSX y guarda respuestas en un XLSX nuevo.")
    p.add_argument("--input", type=Path, default=default_input, help="Spreadsheet .xlsx de entrada")
    p.add_argument("--output", type=Path, default=None, help="Salida .xlsx. Por defecto: Kaggle 1.xlsx, Kaggle 2.xlsx, etc.")
    p.add_argument(
        "--base-url",
        default=os.environ.get("DIFY_BASE_URL", DIFY_BASE_URL_DEFAULT),
        help="Base API.",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("DIFY_API_KEY", DIFY_API_KEY_PLACEHOLDER),
        help="API key.",
    )
    p.add_argument("--delay", type=int, default=5, help="Segundos entre peticiones")
    p.add_argument("--timeout", type=int, default=600, help="Timeout HTTP (s)")
    p.add_argument("--user-id", default="kaggle-xlsx-batch", help="Campo user para Dify")
    args = p.parse_args()

    return run_kaggle_xlsx(args)


if __name__ == "__main__":
    raise SystemExit(main())
