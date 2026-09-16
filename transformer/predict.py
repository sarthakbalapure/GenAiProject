"""
Inference / prediction script for the Document Transformer.

Usage (from project root):
    python -m transformer.predict --image path/to/receipt.jpg
                                  [--box  path/to/box.txt]
                                  [--ckpt path/to/checkpoint_dir]
                                  [--json]

If --box is omitted the script runs Tesseract OCR to obtain word boxes
(requires pytesseract + a Tesseract binary on PATH).
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from PIL import Image

from transformer.config import (
    DEVICE, ID2LABEL, MAX_SEQ_LENGTH, TRANSFORMER_CKPT,
)
from transformer.dataset import get_tokenizer
from transformer.model import DocumentTransformer


# ─── Box loading ─────────────────────────────────────────────────────────────

def _load_boxes_from_file(box_path: str, img_size: Tuple[int, int]):
    """Parse an SROIE-format .txt box file and normalise coordinates."""
    width, height = img_size
    words, boxes = [], []

    with open(box_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", 8)
            if len(parts) < 9:
                continue
            try:
                coords = list(map(int, parts[:8]))
            except ValueError:
                continue
            text = parts[8].strip()
            if not text:
                continue

            xs = coords[0::2]
            ys = coords[1::2]
            x_min = max(0, min(1000, int((min(xs) / width)  * 1000)))
            y_min = max(0, min(1000, int((min(ys) / height) * 1000)))
            x_max = max(0, min(1000, int((max(xs) / width)  * 1000)))
            y_max = max(0, min(1000, int((max(ys) / height) * 1000)))

            if x_min < x_max and y_min < y_max:
                words.append(text)
                boxes.append([x_min, y_min, x_max, y_max])

    return words, boxes


def _load_boxes_from_tesseract(img_path: str):
    """Use pytesseract to obtain word bounding boxes from an image."""
    try:
        import pytesseract
    except ImportError:
        print("[predict] ERROR: pytesseract is not installed. "
              "Install it with: pip install pytesseract", file=sys.stderr)
        sys.exit(1)

    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    tsr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    words, boxes = [], []
    for i, text in enumerate(tsr_data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = int(tsr_data["conf"][i])
        if conf < 0:
            continue

        x = tsr_data["left"][i]
        y = tsr_data["top"][i]
        w = tsr_data["width"][i]
        h = tsr_data["height"][i]

        x_min = max(0, min(1000, int((x          / width)  * 1000)))
        y_min = max(0, min(1000, int((y          / height) * 1000)))
        x_max = max(0, min(1000, int(((x + w)    / width)  * 1000)))
        y_max = max(0, min(1000, int(((y + h)    / height) * 1000)))

        if x_min < x_max and y_min < y_max:
            words.append(text)
            boxes.append([x_min, y_min, x_max, y_max])

    return words, boxes


# ─── Risk / discrepancy layer (rule-based) ───────────────────────────────────

def _run_risk_checks(fields: Dict[str, str]) -> List[str]:
    """
    Deterministic rule-based checks on the extracted fields.
    Returns a list of plain-language flag strings (empty list = no flags).

    NOTE: This is rule logic applied to transformer output, NOT a learned
    risk model.  There is no risk-labelled training data in SROIE/FUNSD.
    """
    import re

    flags = []

    if not fields.get("company"):
        flags.append("MISSING_FIELD: 'company/party name' could not be extracted.")

    if not fields.get("date"):
        flags.append("MISSING_FIELD: 'date' could not be extracted.")
    else:
        # Very basic date-format sanity check (accepts YYYY-MM-DD / DD/MM/YYYY etc.)
        date_val = fields["date"]
        if not re.search(r"\d{1,4}[\-/\.]\d{1,2}[\-/\.]\d{1,4}", date_val):
            flags.append(f"DATE_FORMAT: Extracted date '{date_val}' has unexpected format.")

    if not fields.get("address"):
        flags.append("MISSING_FIELD: 'address' could not be extracted.")

    if not fields.get("total"):
        flags.append("MISSING_FIELD: 'total/amount' could not be extracted.")
    else:
        total_val = fields["total"]
        # Strip currency symbols and whitespace, check that a number remains
        numeric = re.sub(r"[^\d\.,]", "", total_val)
        if not numeric:
            flags.append(f"AMOUNT_INVALID: Extracted total '{total_val}' is non-numeric.")
        elif float(numeric.replace(",", "")) == 0:
            flags.append(f"AMOUNT_ZERO: Extracted total is zero — possible extraction error.")

    return flags


# ─── Main prediction pipeline ─────────────────────────────────────────────────

def predict(
    image_path: str,
    box_path:   str | None = None,
    ckpt_dir:   str | None = None,
) -> Dict:
    """
    End-to-end prediction for a single document image.

    Returns:
        {
            "fields":        {"company": ..., "date": ..., ...},
            "risk_flags":    [...],
            "word_labels":   [{"word": ..., "label": ..., "box": [...]}, ...],
        }
    """
    # ── Load image ─────────────────────────────────────────────────────────
    img = Image.open(image_path).convert("RGB")
    img_size = img.size   # (width, height)

    # ── Get words + boxes ──────────────────────────────────────────────────
    if box_path and os.path.exists(box_path):
        words, boxes = _load_boxes_from_file(box_path, img_size)
    else:
        print("[predict] No box file provided — running Tesseract OCR ...")
        words, boxes = _load_boxes_from_tesseract(image_path)

    if not words:
        return {"error": "No text could be extracted from the document."}

    # ── Tokenise ──────────────────────────────────────────────────────────
    tokenizer = get_tokenizer()
    encoding  = tokenizer(
        words,
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)

    bbox_tensor = torch.tensor([
        boxes[wid] if wid is not None else [0, 0, 0, 0]
        for wid in word_ids
    ], dtype=torch.long).unsqueeze(0)   # (1, seq_len, 4)

    model_input = {
        "input_ids":      encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "token_type_ids": encoding.get("token_type_ids",
                                       torch.zeros_like(encoding["input_ids"])),
        "bbox":           bbox_tensor,
    }

    # ── Load model & run inference ─────────────────────────────────────────
    transformer = DocumentTransformer.load_checkpoint(ckpt_dir)
    fields      = transformer.extract_fields(model_input, words, word_ids)

    # ── Word-level label list (for UI display / debugging) ─────────────────
    logits   = transformer.predict(**model_input)
    pred_ids = logits[0].argmax(dim=-1).cpu().tolist()

    seen = set()
    word_labels = []
    for token_pos, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        word_labels.append({
            "word":  words[wid],
            "label": ID2LABEL[pred_ids[token_pos]],
            "box":   boxes[wid],
        })

    # ── Risk checks ────────────────────────────────────────────────────────
    risk_flags = _run_risk_checks(fields)

    return {
        "fields":      fields,
        "risk_flags":  risk_flags,
        "word_labels": word_labels,
    }


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Transformer field extractor")
    parser.add_argument("--image", required=True, help="Path to document image (jpg/png)")
    parser.add_argument("--box",   default=None,  help="Path to SROIE-format box .txt file")
    parser.add_argument("--ckpt", default=None,   help="Path to fine-tuned checkpoint dir")
    parser.add_argument("--json",  action="store_true", help="Print output as JSON")
    args = parser.parse_args()

    result = predict(args.image, box_path=args.box, ckpt_dir=args.ckpt)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n─── Extracted Fields ───────────────────────────────────")
        for field, value in result.get("fields", {}).items():
            print(f"  {field.upper():12s}: {value or '(not found)'}")

        flags = result.get("risk_flags", [])
        if flags:
            print("\n─── Risk / Discrepancy Flags ───────────────────────────")
            for flag in flags:
                print(f"  ⚠  {flag}")
        else:
            print("\n  ✓  No discrepancies detected.")
