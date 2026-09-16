"""
Training entry point for the Document Transformer (LayoutLM fine-tuning).

Pipeline:
  1. Load & parse SROIE data  (transformer/dataset.py)
  2. Tokenise and align BIO labels
  3. Fine-tune LayoutLM with HuggingFace Trainer
  4. Evaluate with seqeval (precision / recall / F1 per field)
  5. Save best checkpoint to checkpoints/transformer_model/best_model/

Usage (from project root):
    python -m transformer.train
    -- or --
    cd transformer && python train.py
"""

import json
import os

from transformers import (
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)

from transformer.config import (
    BATCH_SIZE,
    DEVICE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    EVAL_STRATEGY,
    LEARNING_RATE,
    LOGGING_STEPS,
    METRIC_FOR_BEST,
    MODEL_NAME,
    OUTPUT_DIR,
    RANDOM_SEED,
    SAVE_TOTAL_LIMIT,
    TRANSFORMER_CKPT,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from transformer.dataset import build_hf_datasets, get_tokenizer
from transformer.evaluate import compute_metrics, evaluate_checkpoint
from transformer.model import DocumentTransformer


def main():
    print("=" * 70)
    print("  Document Transformer — LayoutLM Fine-Tuning on SROIE")
    print(f"  Device: {DEVICE}")
    print(f"  Base model: {MODEL_NAME}")
    print("=" * 70)

    # ── 1. Build datasets ──────────────────────────────────────────────────
    print("\n[1/4] Loading and tokenising SROIE dataset ...")
    train_ds, val_ds = build_hf_datasets()
    print(f"      Train: {len(train_ds):4d} samples  |  Val: {len(val_ds):4d} samples")

    # ── 2. Model ────────────────────────────────────────────────────────────
    print("\n[2/4] Initialising LayoutLM model ...")
    doc_transformer = DocumentTransformer.from_pretrained()
    model     = doc_transformer.model
    tokenizer = get_tokenizer()

    # ── 3. Training setup ───────────────────────────────────────────────────
    print("\n[3/4] Configuring Trainer ...")

    training_args = TrainingArguments(
        output_dir                  = TRANSFORMER_CKPT,
        eval_strategy               = EVAL_STRATEGY,
        save_strategy               = EVAL_STRATEGY,
        learning_rate               = LEARNING_RATE,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        num_train_epochs            = EPOCHS,
        weight_decay                = WEIGHT_DECAY,
        warmup_steps                = int(WARMUP_RATIO * EPOCHS * 100),  # approx 10 % of training
        logging_steps               = LOGGING_STEPS,
        save_total_limit            = SAVE_TOTAL_LIMIT,
        load_best_model_at_end      = True,
        metric_for_best_model       = METRIC_FOR_BEST,
        greater_is_better           = True,
        seed                        = RANDOM_SEED,
        report_to                   = "none",           # Disable wandb/tensorboard by default
        # fp16 = True if DEVICE.type == "cuda" else False,  # Uncomment for A100/V100
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = tokenizer,
        data_collator    = data_collator,
        compute_metrics  = compute_metrics,
        callbacks        = [
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)
        ],
    )

    # ── 4. Train ────────────────────────────────────────────────────────────
    print("\n[4/4] Starting training ...")
    train_result = trainer.train()

    # Log training summary
    print("\n" + "=" * 70)
    print("  Training complete")
    print(f"  Total steps    : {train_result.global_step}")
    print(f"  Train loss     : {train_result.training_loss:.4f}")
    print("=" * 70)

    # ── 5. Save best model ─────────────────────────────────────────────────
    best_model_dir = os.path.join(TRANSFORMER_CKPT, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"\n  ✓  Best model saved → {best_model_dir}")

    # ── 6. Final evaluation ────────────────────────────────────────────────
    metrics = evaluate_checkpoint(trainer, split="val")

    # Save training summary JSON
    summary = {
        "global_step":   train_result.global_step,
        "training_loss": train_result.training_loss,
        "final_metrics": metrics,
    }
    with open(os.path.join(OUTPUT_DIR, "transformer_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  Done!  🎉")


if __name__ == "__main__":
    main()
