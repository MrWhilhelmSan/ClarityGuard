#!/usr/bin/env python3
"""
Envía cada fila de manipulational_conversation.jsonl a la API de chat de Dify
(self-hosted). Usa response_mode=streaming (SSE) porque Agent Chat App no admite blocking.
Cada petición omite conversation_id → conversación nueva.

Cada pregunta al modelo incluye un prefijo en inglés pidiendo respuesta en inglés
y el marco "They sent me this — what's going on?".

Entre una respuesta y la siguiente petición espera DELAY_SECONDS (por defecto 120).

Requiere una instancia self-hosted de Dify con un agent chat App configurado.

Uso:
  cd ~/charlielinux/dify_batch
  python3 dify_batch_chat.py

Menú interactivo: elige bloque (ej. 1–100, 101–200). El estado va en dify_batch_ranges.json.

Sin menú (bloque 3, filas 201–300 si chunk_size=100):
  python3 dify_batch_chat.py --chunk 3

Rango absoluto de filas 1-based (omite menú; estado en row_ranges del JSON de rangos):
  python3 dify_batch_chat.py --from-row 4000 --to-row 4999 --output resp.txt --ranges-file mi_rangos.json --delay 30

Scripts predefinidos (30 s entre peticiones, salida y estado por bloque):
  dify_batch_chat1.py, dify_batch_chat2000.py, dify_batch_chat3000.py (2001–2999),
  dify_batch_chat4000.py … 9000.py

JSONL por defecto: primero en esta carpeta (~/charlielinux/dify_batch), nombre
manipulational_conversation*.jsonl; si no, ~/Downloads y /mnt/c/.../Downloads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Dify API key — configurar via variable de entorno DIFY_API_KEY o editar aqui.
# El valor por defecto es un placeholder; NO uses esta key en produccion.
DIFY_API_KEY_PLACEHOLDER = "<your-dify-api-key>"

DIFY_BASE_URL_DEFAULT = "http://localhost/v1"

# Prefijo enviado al modelo (inglés): respuesta en inglés + contexto "me enviaron esto".
QUERY_PREFIX_EN = (
    "Please reply entirely in English.\n\n"
    "They sent me the following — what's going on?\n\n"
)


def resolve_default_jsonl(script_dir: Path) -> Path:
    """Prioridad: misma carpeta que el script (dify_batch), patrón manipulational_conversation*.jsonl."""
    home = Path.home()
    exact = script_dir / "manipulational_conversation.jsonl"
    if exact.is_file():
        return exact
    matches = sorted(script_dir.glob("manipulational_conversation*.jsonl"))
    if matches:
        return matches[0]
    # NOTA: Los paths siguientes son especificos del autor (entorno WSL + Windows).
    # Si clonas este repo, asegurate de tener el JSONL en el directorio del script
    # o usa --jsonl /ruta/a/tu/archivo.jsonl
    fallback = [
        home / "Downloads" / "manipulational_conversation.jsonl",
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


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_jsonl = resolve_default_jsonl(script_dir)
    default_out = script_dir / "manipulational_conversation_responses.txt"
    default_ranges = script_dir / "dify_batch_ranges.json"

    p = argparse.ArgumentParser(description="Batch Dify chat por bloques desde JSONL.")
    p.add_argument("--jsonl", type=Path, default=None, help="Ruta al .jsonl")
    p.add_argument("--output", type=Path, default=default_out, help="Salida .txt")
    p.add_argument("--ranges-file", type=Path, default=default_ranges, help="Estado de bloques (completado / en curso)")
    p.add_argument("--chunk-size", type=int, default=100, help="Filas por bloque (ej. 100 → 1–100, 101–200)")
    p.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Número de bloque 1-based sin menú (ej. 2 = filas 101–200 con chunk-size 100). 0 = menú interactivo.",
    )
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
    p.add_argument("--user-id", default="jsonl-batch", help="Campo user para Dify")
    p.add_argument(
        "--redo",
        action="store_true",
        help="Volver a procesar un bloque aunque ya esté marcado como completo",
    )
    p.add_argument(
        "--from-row",
        type=int,
        default=None,
        metavar="N",
        help="Fila inicial 1-based (inclusive). Con --to-row omite el menú y procesa solo ese rango.",
    )
    p.add_argument(
        "--to-row",
        type=int,
        default=None,
        metavar="N",
        help="Fila final 1-based (inclusive). Requiere --from-row.",
    )
    args = p.parse_args()

    jsonl_path = args.jsonl if args.jsonl is not None else default_jsonl

    if not args.api_key:
        print("Falta API key.", file=sys.stderr)
        return 1

    if not jsonl_path.is_file():
        print(f"No existe el JSONL: {jsonl_path}", file=sys.stderr)
        print(
            f"Pon el archivo en la carpeta del script ({script_dir}) "
            "como manipulational_conversation.jsonl o manipulational_conversation*.jsonl, "
            "o usa --jsonl /ruta/archivo.jsonl",
            file=sys.stderr,
        )
        return 1

    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Línea {line_no}: JSON inválido: {e}", file=sys.stderr)
                return 1

    total = len(records)
    chunk_size = max(1, args.chunk_size)
    num_chunks = math.ceil(total / chunk_size) if total else 0

    state = load_ranges_state(args.ranges_file)
    if state.get("chunk_size") and state["chunk_size"] != chunk_size:
        print(
            f"Aviso: chunk_size en {args.ranges_file} era {state.get('chunk_size')}, ahora {chunk_size}. "
            "Los índices de bloque pueden no coincidir con corridas antiguas.",
            file=sys.stderr,
        )

    print(f"JSONL: {jsonl_path}")

    if args.from_row is not None or args.to_row is not None:
        if args.from_row is None or args.to_row is None:
            print("Debes indicar ambos: --from-row y --to-row.", file=sys.stderr)
            return 1
        return run_row_range(
            start_1=args.from_row,
            end_1=args.to_row,
            records=records,
            state=state,
            ranges_path=args.ranges_file,
            args=args,
            redo=args.redo,
        )

    chunk_0: int | None
    if args.chunk > 0:
        if args.chunk > num_chunks:
            print(f"No existe el bloque {args.chunk} (solo hay {num_chunks} bloques).", file=sys.stderr)
            return 1
        chunk_0 = args.chunk - 1
        if is_chunk_fully_done(chunk_0, chunk_size, total, state) and not args.redo:
            start_1, end_1 = chunk_row_bounds(chunk_0, chunk_size, total)
            print(
                f"El bloque {args.chunk} (filas {start_1}–{end_1}) YA está procesado por completo. "
                "Usa --redo para repetir o elige otro con --chunk.",
                file=sys.stderr,
            )
            return 1
    else:
        chunk_0 = pick_chunk_interactive(num_chunks, chunk_size, total, state, args.redo)
        if chunk_0 is None:
            print("Saliendo.")
            return 0

    return run_chunk(
        chunk_index_0=chunk_0,
        chunk_size=chunk_size,
        records=records,
        state=state,
        ranges_path=args.ranges_file,
        args=args,
        redo=args.redo,
    )


if __name__ == "__main__":
    raise SystemExit(main())
