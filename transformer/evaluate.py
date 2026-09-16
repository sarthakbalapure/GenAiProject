"""
seqeval-based evaluation metrics for the Document Transformer.

Computes per-class and overall precision / recall / F1.
Compatible with HuggingFace Trainer's `compute_metrics` callback.
"""

import json
import os
from typing import Dict, Tuple

import numpy as np
import evaluate as hf_evaluate

from transformer.config import LABELS, OUTPUT_DIR


# ─── Load seqeval once at module import ──────────────────────────────────────
_seqeval = hf_evaluate.load("seqeval")


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    HuggingFace Trainer-compatible compute_metrics function.
    Strips -100 (padding/special tokens) before evaluation.
    """
    logits, label_ids = eval_pred
    predictions = np.argmax(logits, axis=2)  # (B, seq_len)

    true_preds  = [
        [LABELS[p] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, label_ids)
    ]
    true_labels = [
        [LABELS[l] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, label_ids)
    ]

    results = _seqeval.compute(predictions=true_preds, references=true_labels)

    # Flatten per-entity-class F1 into the returned dict for easy inspection
    per_class = {
        f"{entity}_f1": round(v["f1"], 4)
        for entity, v in results.items()
        if isinstance(v, dict) and "f1" in v
    }

    return {
        "precision": round(results["overall_precision"], 4),
        "recall":    round(results["overall_recall"],    4),
        "f1":        round(results["overall_f1"],        4),
        "accuracy":  round(results["overall_accuracy"],  4),
        **per_class,
    }


def evaluate_checkpoint(trainer, split: str = "val") -> Dict[str, float]:
    """
    Run full evaluation on the given trainer's eval dataset and save results to disk.
    Returns the metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"  Final evaluation on {split} set")
    print(f"{'='*60}")
    metrics = trainer.evaluate()
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s}: {v}")

    out_path = os.path.join(OUTPUT_DIR, f"transformer_{split}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {out_path}")
    return metrics
