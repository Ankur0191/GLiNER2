"""Train a small domain adapter from reviewed GLiNER2 JSONL annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gliner2 import AutoExtractor
from gliner2.training.config import TrainingConfig
from gliner2.training.trainer import ExtractorTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="fastino/gliner2.5-base-v1")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--full-finetune", action="store_true")
    args = parser.parse_args()

    train_path = Path(args.train)
    validation_path = Path(args.validation) if args.validation else None
    if not train_path.is_file():
        parser.error(f"Training file not found: {train_path}")
    if validation_path and not validation_path.is_file():
        parser.error(f"Validation file not found: {validation_path}")

    model = AutoExtractor.from_pretrained(args.base_model)
    config = TrainingConfig(
        output_dir=args.output_dir,
        experiment_name=f"rivalmap-{Path(args.output_dir).name}",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        encoder_lr=args.learning_rate,
        task_lr=5e-4,
        eval_strategy="epoch" if validation_path else "no",
        save_best=bool(validation_path),
        early_stopping=bool(validation_path),
        early_stopping_patience=2,
        validate_data=True,
        use_lora=not args.full_finetune,
        lora_r=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
        lora_target_modules=["encoder"],
        save_adapter_only=not args.full_finetune,
        fp16=False,
        bf16=False,
    )
    trainer = ExtractorTrainer(model, config)
    report = trainer.train(train_data=str(train_path), eval_data=str(validation_path) if validation_path else None)
    Path(args.output_dir, "training_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
