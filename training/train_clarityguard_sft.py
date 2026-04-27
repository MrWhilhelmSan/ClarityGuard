#!/usr/bin/env python3
"""
Fine-tuning ClarityGuard con Unsloth (QLoRA 4-bit) + TRL SFTTrainer.
Dataset: JSONL chat (user/assistant), ver prepare_dataset.py.

Ejemplos:
  python train_clarityguard_sft.py --config train_config.yaml --profile pilot_1k --dry-run
  python train_clarityguard_sft.py --config train_config.yaml --profile pilot_1k
  python train_clarityguard_sft.py --config train_config.yaml --profile scale_10k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from prepare_dataset import (
    load_jsonl_messages,
    prepare_messages_row,
    validate_dataset,
)


def load_merged_config(config_path: Path, profile_name: str) -> dict:
    with config_path.open(encoding="utf-8") as f:
        full = yaml.safe_load(f)
    if "defaults" not in full or "profiles" not in full:
        raise ValueError("train_config.yaml debe contener 'defaults' y 'profiles'.")
    if profile_name not in full["profiles"]:
        raise KeyError(f"Perfil desconocido: {profile_name}. Disponibles: {list(full['profiles'])}")
    defaults = dict(full["defaults"])
    prof = dict(full["profiles"][profile_name])
    return {**defaults, **prof}


def resolve_path(config_path: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (config_path.resolve().parent / p).resolve()


def dry_run(config_path: Path, profile_name: str) -> int:
    cfg = load_merged_config(config_path, profile_name)
    data_path = resolve_path(config_path, cfg["data_file"])
    print("Perfil:", profile_name)
    print("model_name:", cfg["model_name"])
    print("data_file:", data_path)
    if not data_path.is_file():
        print("ERROR: no existe el fichero de datos.", file=sys.stderr)
        return 1
    rows = load_jsonl_messages(data_path)
    stats = validate_dataset(rows)
    print("Validación dataset:", stats)
    sample = prepare_messages_row(
        rows[0],
        strip_dataset_id_line=cfg.get("strip_dataset_id_line", True),
        system_prompt=cfg.get("system_prompt"),
    )
    print("Primer ejemplo (roles tras preparar):", [m["role"] for m in sample["messages"]])
    print("Dry-run OK.")
    return 0


def run_train(config_path: Path, profile_name: str) -> int:
    cfg = load_merged_config(config_path, profile_name)
    data_path = resolve_path(config_path, cfg["data_file"])
    output_dir = resolve_path(config_path, cfg["output_dir"])

    if not data_path.is_file():
        print(f"ERROR: no existe data_file: {data_path}", file=sys.stderr)
        return 1

    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
    except ImportError:
        print(
            "Instala Unsloth y dependencias (ver requirements-training.txt o unsloth.ai/docs).",
            file=sys.stderr,
        )
        return 1

    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    rows = load_jsonl_messages(data_path)
    stats = validate_dataset(rows)
    if stats["bad_rows"]:
        print(f"Advertencia: filas con formato inesperado: {stats['bad_rows']}", file=sys.stderr)

    prepared = [
        prepare_messages_row(
            r,
            strip_dataset_id_line=cfg.get("strip_dataset_id_line", True),
            system_prompt=cfg.get("system_prompt"),
        )
        for r in rows
    ]
    raw_ds = Dataset.from_list(prepared)

    max_seq_length = int(cfg["max_seq_length"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg["lora_r"]),
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        use_gradient_checkpointing="unsloth" if cfg.get("gradient_checkpointing") else False,
        random_state=int(cfg.get("seed", 3407)),
        use_rslora=False,
        loftq_config=None,
    )

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conv in examples["messages"]
        ]
        return {"text": texts}

    dataset = raw_ds.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=raw_ds.column_names,
    )

    val_ratio = float(cfg.get("val_ratio", 0.0))
    eval_dataset = None
    if val_ratio > 0 and len(dataset) > 5:
        split = dataset.train_test_split(test_size=val_ratio, seed=int(cfg.get("seed", 3407)))
        dataset = split["train"]
        eval_dataset = split["test"]

    use_bf16 = bool(cfg.get("bf16", True)) and is_bfloat16_supported()
    ta_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        learning_rate=float(cfg["learning_rate"]),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        save_total_limit=3,
        seed=int(cfg.get("seed", 3407)),
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        report_to="none",
    )
    if eval_dataset is not None:
        ta_kwargs["eval_strategy"] = "steps"
        ta_kwargs["eval_steps"] = int(cfg["save_steps"])
    train_args = TrainingArguments(**ta_kwargs)

    max_steps = cfg.get("max_steps", -1)
    if max_steps is not None and int(max_steps) > 0:
        train_args.max_steps = int(max_steps)
    else:
        train_args.num_train_epochs = float(cfg["num_train_epochs"])

    trainer_kwargs = dict(
        model=model,
        train_dataset=dataset,
        args=train_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        dataset_num_proc=1,
    )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("tokenizer", None)
        trainer_kwargs["processing_class"] = tokenizer
        trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if cfg.get("merge_and_save_16bit"):
        merged = output_dir / "merged_16bit"
        merged.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained_merged(
                str(merged),
                tokenizer,
                save_method="merged_16bit",
            )
            print("Modelo fusionado guardado en:", merged)
        except Exception as e:
            print("No se pudo guardar merge 16-bit (opcional):", e, file=sys.stderr)

    print("Entrenamiento terminado. Artefactos en:", output_dir)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "train_config.yaml")
    ap.add_argument("--profile", type=str, default="pilot_1k")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run(args.config, args.profile)
    return run_train(args.config, args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
