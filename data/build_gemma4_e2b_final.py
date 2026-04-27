#!/usr/bin/env python3
"""
Build final SFT JSONL for Gemma 4 E2B training.
Aligned with ClarityGuard Assistant v2.1 prompt rules.

Key rules from the prompt:
- NEVER use: "manipulation", "gaslighting", "victim", "aggressor", "abuser"
  → Replace with pattern-descriptive language
- All analysis visible (no think blocks) for 2B model
- No dataset metadata in user prompts
- English throughout (consistent language)
- C.F.R.V.A. scoring with /10 per dimension, /50 total
- Structure: Analysis → Defense → Suggestion → Education → Follow-up
- Gemma 4 text format (non-thinking template)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
DATASET_META = re.compile(r"\[dataset_id=[^\]]*\]")
MULTI_NEWLINE = re.compile(r"\n{3,}")

# ─── FORBIDDEN TERM REPLACEMENTS ───
# Per ClarityGuard v2.1: NEVER use "manipulation", "gaslighting", etc.
# Replace with pattern-descriptive language as the prompt specifies.
FORBIDDEN_REPLACEMENTS = [
    # Exact phrases first (longer matches before shorter)
    ("manipulative communication", "problematic communication pattern"),
    ("manipulative behavior", "problematic behavioral pattern"),
    ("manipulative pattern", "problematic communication pattern"),
    ("manipulative tactic", "unclear communication tactic"),
    ("manipulative intent", "unclear communicative intent"),
    ("manipulative technique", "problematic communication technique"),
    ("manipulative strategy", "problematic communication strategy"),
    ("manipulative language", "structurally unclear language"),
    ("manipulative dynamic", "problematic relational dynamic"),
    ("MANIPULATIVE COMMUNICATION", "PROBLEMATIC COMMUNICATION PATTERN"),
    ("Manipulative Communication", "Problematic Communication Pattern"),
    ("manipulate the situation", "control the narrative"),
    ("manipulate you", "influence your response"),
    ("manipulatively", "through unclear communication"),
    ("manipulating", "using unclear communication patterns on"),
    ("manipulators", "sources of problematic patterns"),
    ("manipulator", "source of problematic patterns"),
    ("Manipulative", "Problematic"),
    ("manipulative", "problematic"),
    ("MANIPULATIVE", "PROBLEMATIC"),
    ("MANIPULATIVA", "PROBLEMATIC"),
    ("MANIPULATIVO", "PROBLEMATIC"),
    ("manipulate", "influence through ambiguity"),
    ("manipulation", "problematic communication pattern"),
    ("Manipulation", "Problematic Communication Pattern"),
    ("MANIPULATION", "PROBLEMATIC COMMUNICATION PATTERN"),
    ("Gaslighters", "Sources of reality-distortion"),
    ("gaslighters", "sources of reality-distortion"),
    ("Gaslighter", "Source of reality-distortion"),
    ("gaslighter", "source of reality-distortion"),
    ("Gaslights", "Distorts reality"),
    ("gaslights", "distorts reality"),
    ("gaslighting pattern", "reality-distortion pattern"),
    ("gaslighting technique", "reality-distortion technique"),
    ("gaslighting", "reality-distortion pattern"),
    ("Gaslighting", "Reality-Distortion Pattern"),
    ("GASLIGHTING", "REALITY-DISTORTION PATTERN"),
    ("Gaslight", "Reality-distortion"),
    ("gaslight", "reality-distortion"),
    ("MANIPULACIÓN", "PROBLEMATIC COMMUNICATION PATTERN"),
    ("Manipulación", "Problematic Communication Pattern"),
    ("victim/burden-bearer", "affected party"),
    ("victim mentality", "self-positioning as affected party"),
    ("victim", "affected party"),
    ("Victim", "Affected Party"),
    ("VICTIM", "AFFECTED PARTY"),
    ("aggressor", "source of the pattern"),
    ("Aggressor", "Source of the Pattern"),
    ("AGGRESSOR", "SOURCE OF THE PATTERN"),
    ("abuser", "pattern source"),
    ("Abuser", "Pattern Source"),
    ("ABUSER", "PATTERN SOURCE"),
    # Spanish equivalents
    ("manipulación", "patrón de comunicación problemático"),
    ("Manipulación", "Patrón de Comunicación Problemático"),
    ("víctima", "parte afectada"),
    ("agresor", "fuente del patrón"),
    ("abusador", "fuente del patrón"),
]

# ─── SPANISH → ENGLISH NORMALIZATION ───
LANG_REPLACEMENTS = [
    ("Juez 1", "Judge 1"),
    ("Juez 2", "Judge 2"),
    ("Juez 3", "Judge 3"),
    ("JUEZ 1", "JUDGE 1"),
    ("JUEZ 2", "JUDGE 2"),
    ("JUEZ 3", "JUDGE 3"),
    ("Detectado:", "Detected:"),
    ("Puntaje:", "Score:"),
    ("DEFENSA ASERTIVA DIRECTA", "DIRECT ASSERTIVE DEFENSE"),
    ("PROTECCIÓN COGNITIVA", "COGNITIVE PROTECTION"),
    ("SUGERENCIA DE ACCIÓN ASERTIVA", "SUGGESTED ASSERTIVE ACTION"),
    ("SUGERENCIA DE COMUNICACIÓN ASERTIVA", "SUGGESTED ASSERTIVE COMMUNICATION"),
    ("EDUCACIÓN SOBRE PATRONES", "EDUCATION ON PATTERNS"),
    ("RESISTENCIA ESPERADA", "EXPECTED RESISTANCE"),
    ("PLAN DE FOLLOW-UP ITERATIVO", "ITERATIVE FOLLOW-UP PLAN"),
    ("PLAN DE SEGUIMIENTO", "FOLLOW-UP PLAN"),
    ("EXIGIR CLARIDAD", "DEMANDING CLARITY"),
    ("EXIGING CLARITY", "DEMANDING CLARITY"),
    ("CONTEXTO NO DECLARADO", "CONTEXT NOT DECLARED"),
    ("Contexto No Declarado", "Context Not Declared"),
    ("Contexto no declarado", "Context not declared"),
    ("FALTA DE CLARIDAD OPERATIVA", "LACK OF OPERATIONAL CLARITY"),
    ("Falta de Claridad Operativa", "Lack of Operational Clarity"),
    ("Falta de claridad operativa", "Lack of operational clarity"),
    ("REDIRECCIÓN TEMÁTICA", "THEMATIC REDIRECTION"),
    ("Redirección Temática", "Thematic Redirection"),
    ("Redirección temática", "Thematic redirection"),
    ("VALIDACIÓN CONDICIONAL", "CONDITIONAL VALIDATION"),
    ("Validación Condicional", "Conditional Validation"),
    ("Validación condicional", "Conditional validation"),
    ("FALTA DE ACCESIBILIDAD COMUNICATIVA", "LACK OF COMMUNICATIVE ACCESSIBILITY"),
    ("FALTA DE ACCESIBILIDAD", "LACK OF ACCESSIBILITY"),
    ("Falta de Accesibilidad", "Lack of Accessibility"),
    ("Falta de accesibilidad", "Lack of accessibility"),
    ("ARQUITECTURA DE FALTA DE CLARIDAD", "ARCHITECTURE OF LACK OF CLARITY"),
    ("ARQUITECTURA DE CLARIDAD", "CLARITY ARCHITECTURE"),
    ("Arquitectura de Falta de Claridad", "Architecture of Lack of Clarity"),
    ("Arquitectura de Claridad", "Clarity Architecture"),
    ("Coherencia Comunicativa", "Communicative Coherence"),
    ("COHERENCIA COMUNICATIVA", "COMMUNICATIVE COHERENCE"),
    ("Coherencia Factual", "Factual Coherence"),
    ("COHERENCIA FACTUAL", "FACTUAL COHERENCE"),
    ("CLARIDAD OPERATIVA", "OPERATIONAL CLARITY"),
    ("CLARIDAD", "CLARITY"),
    ("ACCESIBILIDAD COMUNICATIVA", "COMMUNICATIVE ACCESSIBILITY"),
    ("ACCESIBILIDAD RESPONSIVA", "RESPONSIVE ACCESSIBILITY"),
    ("ACCESIBILIDAD", "ACCESSIBILITY"),
    ("COMUNICACIÓN ASERTIVA", "ASSERTIVE COMMUNICATION"),
    ("COMUNICACIÓN", "COMMUNICATION"),
    ("REDIRECCIÓN", "REDIRECTION"),
    ("PUNTAJE TOTAL", "TOTAL SCORE"),
    ("Puntaje total", "Total score"),
    ("Puntaje Total", "Total Score"),
    ("Hechos claros:", "Clear facts:"),
    ("Puntaje C.F.R.V.A.", "C.F.R.V.A. Score"),
    ("Puntaje C.C.R.V.A.", "C.C.R.V.A. Score"),
    ("Puntaje:", "Score:"),
    ("comunicación problemática", "problematic communication"),
    ("Conclusión:", "Conclusion:"),
    ("conclusión", "conclusion"),
    ("Resumen:", "Summary:"),
    ("Resultado:", "Result:"),
    ("Hallazgo:", "Finding:"),
]


def clean_user_prompt(text: str) -> str:
    text = DATASET_META.sub("", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def extract_full_visible_response(assistant: str) -> str:
    if THINK_CLOSE in assistant:
        public = assistant.split(THINK_CLOSE, 1)[1].strip()
        if len(public) > 200:
            return public
        text = assistant.replace(THINK_OPEN, "").replace(THINK_CLOSE, "").strip()
        return text
    return assistant.strip()


def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def clean_assistant(text: str) -> str:
    text = apply_replacements(text, FORBIDDEN_REPLACEMENTS)
    text = apply_replacements(text, LANG_REPLACEMENTS)
    text = DATASET_META.sub("", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def is_predominantly_spanish(text: str) -> bool:
    es_words = sum(1 for w in ["una", "que", "del", "con", "por", "los", "las",
                                "este", "esta", "como", "pero", "sobre", "está",
                                "puede", "para", "más", "ser"]
                   if f" {w} " in text.lower())
    en_words = sum(1 for w in ["the", "and", "you", "your", "this", "that", "with",
                                "for", "not", "but", "what", "from", "are", "was"]
                   if f" {w} " in text.lower())
    return es_words > en_words


def build_gemma_text(user: str, assistant: str) -> str:
    return (
        f"<|turn>user\n{user}\n<turn|>\n"
        f"<|turn>model\n{assistant}\n<turn|>\n"
    )


def main() -> None:
    src = Path("/mnt/c/Users/carlo/Documents/clean/gemma4_unsloth_sft_chat1_5000.jsonl")
    dst = Path("/mnt/c/Users/carlo/Documents/clean/gemma4_e2b_sft_ready.jsonl")

    stats = Counter()

    with dst.open("w", encoding="utf-8") as fout:
        for line in src.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            msgs = o.get("messages") or o.get("conversations")
            if not msgs:
                stats["skip_no_msgs"] += 1
                continue

            user_msg = [m for m in msgs if m["role"] == "user"]
            asst_msg = [m for m in msgs if m["role"] == "assistant"]
            if not user_msg or not asst_msg:
                stats["skip_missing_role"] += 1
                continue

            user_text = clean_user_prompt(user_msg[0]["content"])
            asst_text = extract_full_visible_response(asst_msg[0]["content"])
            asst_text = clean_assistant(asst_text)

            if len(asst_text) < 300:
                stats["skip_short_response"] += 1
                continue

            if is_predominantly_spanish(asst_text):
                stats["skip_spanish"] += 1
                continue

            new_messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": asst_text},
            ]

            out = {
                "conversation_id": o.get("conversation_id"),
                "manipulation_type": o.get("manipulation_type"),
                "is_manipulation": o.get("is_manipulation"),
                "context_type": o.get("context_type"),
                "messages": new_messages,
                "text": build_gemma_text(user_text, asst_text),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["written"] += 1

    # Print stats
    types = Counter()
    manip = Counter()
    for line in dst.open(encoding="utf-8"):
        o = json.loads(line)
        types[o.get("manipulation_type")] += 1
        manip[o.get("is_manipulation")] += 1

    print(json.dumps({
        "output_file": str(dst),
        "stats": dict(stats),
        "manipulation_types": dict(types.most_common()),
        "is_manipulation": dict(manip),
        "total_rows": stats["written"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
