#!/usr/bin/env python3
"""
Build SFT JSONL: full judge reasoning stays inside <think>...</think>;
public assistant text strips evaluators / C.F.R.V.A. / ClarityGuard scaffolding.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"

DATASET_LINE = re.compile(r"\n\n\[dataset_id=[^\]]+\]\n\n", re.MULTILINE)

# Remove branding / framework names from visible tail
SCRUB_PATTERNS = [
    (re.compile(r"\*\*\[ClarityGuard\]\s*Analysis Detected:\s*", re.I), "**What stands out:** "),
    (re.compile(r"\[ClarityGuard\]\s*", re.I), ""),
    (re.compile(r"ClarityGuard\s*", re.I), ""),
    (re.compile(r"^.*C\.F\.R\.V\.A\.\s*Score:.*\n", re.MULTILINE | re.I), ""),
    (re.compile(r"^.*Total Score:.*\n", re.MULTILINE | re.I), ""),
    (re.compile(r"^.*\bJudge\s*\d.*\n", re.MULTILINE | re.I), ""),
    (re.compile(r"^.*\bJuez\s*\d.*\n", re.MULTILINE | re.I), ""),
    (re.compile(r"^.*C\.F\.R\.V\.A\..*\n", re.MULTILINE | re.I), ""),
    (re.compile(r"^.*three evaluators.*\n", re.MULTILINE | re.I), ""),
    (re.compile(r"^.*internal evaluators.*\n", re.MULTILINE | re.I), ""),
]


def clean_user_content(text: str) -> str:
    return DATASET_LINE.sub("\n\n", text)


def split_think(assistant: str) -> tuple[str, str]:
    if THINK_OPEN not in assistant or THINK_CLOSE not in assistant:
        raise ValueError("assistant missing think delimiters")
    after_open = assistant.split(THINK_OPEN, 1)[1]
    inner, visible = after_open.split(THINK_CLOSE, 1)
    return inner.strip(), visible.lstrip()


def first_content_index(pub: str) -> int | None:
    idxs: list[int] = []
    for needle in ("### 🔒", "### ✍️"):
        i = pub.find(needle)
        if i != -1:
            idxs.append(i)
    j = pub.find("**Your feeling")
    if j != -1:
        idxs.append(j)
    return min(idxs) if idxs else None


def cut_trailing_modules(pub: str) -> str:
    stops = [
        "\n### 🛡️",
        "\n### ⏰",
        "\n### 📋",
        "\n### 🎯",
        "\n## 🛡️",
    ]
    best = len(pub)
    for s in stops:
        i = pub.find(s)
        if i != -1 and i < best:
            best = i
    return pub[:best].rstrip()


def extract_user_facing_visible(pub: str) -> str:
    """Keep defense + suggested-action style blocks; drop judge exposition."""
    p = pub
    cut = first_content_index(p)
    if cut is not None:
        p = p[cut:]

    p = cut_trailing_modules(p)

    # Drop orphan intro lines before first ### if any remain (rare)
    lines = p.split("\n")
    out: list[str] = []
    seen_header = False
    for ln in lines:
        if ln.startswith("### "):
            seen_header = True
        if not seen_header and re.match(r"^[🔍✨💡]?\s*\*\*What stands out", ln):
            seen_header = True
        if not seen_header and re.search(
            r"Judge|Juez|C\.F\.R\.V\.A|evaluators|ClarityGuard", ln, re.I
        ):
            continue
        if not seen_header and ln.strip() and not ln.startswith("#"):
            if re.search(r"Score:\s*\d|Problematic Communication|breakdown", ln, re.I):
                continue
        out.append(ln)
    p = "\n".join(out).strip()

    for rx, rep in SCRUB_PATTERNS:
        p = rx.sub(rep, p)

    # Remove leftover blank-padded duplicate headers
    p = re.sub(r"\n{3,}", "\n\n", p).strip()

    # Final pass: drop any line that still names the framework acronym
    p = "\n".join(
        ln for ln in p.split("\n") if not re.search(r"C\.F\.R\.V\.A", ln, re.I)
    )
    p = re.sub(r"\n{3,}", "\n\n", p).strip()

    if len(p) < 80:
        # Minimal fallback: strip obvious judge lines from original visible
        raw_lines = pub.split("\n")
        kept = [
            ln
            for ln in raw_lines
            if not re.search(
                r"Juez\s*\d|Judge\s*\d|C\.F\.R\.V\.A\.|three evaluators", ln, re.I
            )
        ]
        p = "\n".join(kept).strip()
        p = cut_trailing_modules(p)
        for rx, rep in SCRUB_PATTERNS:
            p = rx.sub(rep, p)
        p = re.sub(r"\n{3,}", "\n\n", p).strip()

    return p


def build_assistant(inner: str, public_short: str) -> str:
    return f"{THINK_OPEN}\n{inner}\n{THINK_CLOSE}\n{public_short}"


def build_gemma_text(user: str, assistant: str) -> str:
    return (
        f"<|turn>user\n{user}\n<turn|>\n"
        f"<|turn>model\n{assistant}\n<turn|>\n"
    )


def main() -> None:
    src = Path(__file__).resolve().parent / "gemma4_unsloth_sft.jsonl"
    dst = Path(__file__).resolve().parent / "gemma4_unsloth_sft_internalize.jsonl"

    stats = {"n": 0, "short_len_min": 10**9, "short_len_max": 0, "fallback_short": 0}

    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            msgs = o.get("messages") or o.get("conversations")
            if not msgs:
                continue

            new_msgs = []
            for m in msgs:
                if m["role"] == "user":
                    new_msgs.append(
                        {"role": "user", "content": clean_user_content(m["content"])}
                    )
                elif m["role"] == "assistant":
                    inner, visible = split_think(m["content"])
                    pub = extract_user_facing_visible(visible)
                    if len(pub) < 80:
                        stats["fallback_short"] += 1
                    stats["short_len_min"] = min(stats["short_len_min"], len(pub))
                    stats["short_len_max"] = max(stats["short_len_max"], len(pub))
                    new_msgs.append(
                        {"role": "assistant", "content": build_assistant(inner, pub)}
                    )
                else:
                    new_msgs.append(dict(m))

            out = {
                "conversation_id": o.get("conversation_id"),
                "manipulation_type": o.get("manipulation_type"),
                "is_manipulation": o.get("is_manipulation"),
                "context_type": o.get("context_type"),
                "messages": new_msgs,
                "text": build_gemma_text(new_msgs[0]["content"], new_msgs[1]["content"]),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["n"] += 1

    if stats["short_len_min"] == 10**9:
        stats["short_len_min"] = 0
    print(json.dumps({"output": str(dst), **stats}, indent=2))


if __name__ == "__main__":
    main()
