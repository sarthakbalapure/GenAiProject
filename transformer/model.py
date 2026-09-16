"""
LayoutLM model wrapper for token-level field extraction.

Thin wrapper around HuggingFace AutoModelForTokenClassification that adds:
  - Convenience factory (from_pretrained / load_checkpoint)
  - Parameter count helper
  - Inference helper returning structured field dicts
"""

from __future__ import annotations

import os
from typing import Dict, List

import torch
from transformers import AutoModelForTokenClassification

from transformer.config import (
    MODEL_NAME, NUM_LABELS, ID2LABEL, LABEL2ID, DEVICE, TRANSFORMER_CKPT,
)


class DocumentTransformer:
    """
    Wraps a HuggingFace LayoutLM model with project-specific helpers.
    The underlying `model` attribute is a standard
    `AutoModelForTokenClassification` and is fully compatible with HF Trainer.
    """

    def __init__(self, model: AutoModelForTokenClassification):
        self.model = model

    # ─── Factory methods ─────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls) -> "DocumentTransformer":
        """
        Load the base pretrained LayoutLM weights from HuggingFace Hub
        and attach our NER classification head.
        """
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,   # classification head is new
        )
        model.to(DEVICE)
        print(f"[Transformer] Loaded '{MODEL_NAME}' — {_count_params(model):,} trainable params")
        return cls(model)

    @classmethod
    def load_checkpoint(cls, ckpt_dir: str | None = None) -> "DocumentTransformer":
        """
        Load a fine-tuned checkpoint from disk.
        Falls back to TRANSFORMER_CKPT / best_model if ckpt_dir is None.
        """
        path = ckpt_dir or os.path.join(TRANSFORMER_CKPT, "best_model")
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"[Transformer] Checkpoint not found at '{path}'. "
                "Run train.py first to produce a checkpoint."
            )
        model = AutoModelForTokenClassification.from_pretrained(path)
        model.to(DEVICE)
        print(f"[Transformer] Loaded checkpoint from '{path}'")
        return cls(model)

    # ─── Inference ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        bbox: torch.Tensor,
    ) -> torch.Tensor:
        """
        Raw forward pass.  Returns logits of shape (B, seq_len, num_labels).
        """
        self.model.eval()
        outputs = self.model(
            input_ids=input_ids.to(DEVICE),
            attention_mask=attention_mask.to(DEVICE),
            token_type_ids=token_type_ids.to(DEVICE),
            bbox=bbox.to(DEVICE),
        )
        return outputs.logits  # (B, seq_len, num_labels)

    @torch.no_grad()
    def extract_fields(
        self,
        tokenized_input: Dict[str, torch.Tensor],
        words: List[str],
        word_ids: List[int | None],
    ) -> Dict[str, str]:
        """
        High-level extraction: runs inference and returns a dict mapping
        each detected field name to the concatenated text span.

        Args:
            tokenized_input: dict from tokenizer (input_ids, attention_mask,
                             token_type_ids, bbox) — all unsqueezed to batch=1.
            words:           original word list (before subword tokenisation).
            word_ids:        word index for each token position (from tokenizer).

        Returns:
            {
                "company": "...",
                "date":    "...",
                "address": "...",
                "total":   "...",
            }
        """
        logits   = self.predict(**tokenized_input)           # (1, seq_len, C)
        pred_ids = logits[0].argmax(dim=-1).cpu().tolist()   # (seq_len,)

        # Map token predictions back to word-level (majority vote: first B-/I- wins)
        word_labels: Dict[int, str] = {}
        for token_pos, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in word_labels:
                continue
            label = ID2LABEL[pred_ids[token_pos]]
            if label != "O":
                word_labels[word_idx] = label

        # Group contiguous word spans by field
        fields: Dict[str, List[str]] = {
            "company": [], "date": [], "address": [], "total": [],
        }
        for word_idx in sorted(word_labels):
            label     = word_labels[word_idx]
            field_key = label.split("-")[-1].lower()  # e.g. "COMPANY" → "company"
            if field_key in fields:
                fields[field_key].append(words[word_idx])

        return {k: " ".join(v) for k, v in fields.items()}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
