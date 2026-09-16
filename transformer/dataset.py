"""
SROIE Dataset loading and tokenization for LayoutLM.

Parses the SROIE box/key annotations, assigns BIO NER tags to each OCR word,
then tokenizes and aligns labels for subword tokenization (WordPiece).
"""

import os
import json
import random
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, Features, Sequence, ClassLabel, Value
from transformers import AutoTokenizer

from transformer.config import (
    BOX_DIR, IMG_DIR, KEY_DIR,
    LABELS, LABEL2ID,
    MODEL_NAME, MAX_SEQ_LENGTH,
    VAL_RATIO, RANDOM_SEED,
)


# ─── Tag assignment ───────────────────────────────────────────────────────────

def _assign_ner_tags(words: List[str], keys: Dict[str, str]) -> List[str]:
    """
    Match each OCR word to a key field using substring lookup and assign BIO tags.

    Strategy:
    1. For each field (company/date/address/total), tokenize the value by
       whitespace and check whether the word sequence appears in the OCR words.
    2. Assign B- to the first match token, I- to subsequent ones.
    3. Everything else is "O".
    """
    tags = ["O"] * len(words)

    for key, value in keys.items():
        if not value or not isinstance(value, str):
            continue

        key_upper = key.upper()
        if key_upper not in [l[2:] for l in LABELS if l.startswith("B-")]:
            continue  # Ignore unknown field keys

        val_tokens = value.strip().split()
        if not val_tokens:
            continue

        # Slide a window over OCR words to find the best contiguous match
        for start_idx in range(len(words) - len(val_tokens) + 1):
            # Check if each value token is a substring of the corresponding word
            match = all(
                vt.lower() in words[start_idx + j].lower()
                for j, vt in enumerate(val_tokens)
            )
            if match:
                tags[start_idx] = f"B-{key_upper}"
                for j in range(1, len(val_tokens)):
                    tags[start_idx + j] = f"I-{key_upper}"
                break  # Take first contiguous match only

    return tags


# ─── Single-file parsing ─────────────────────────────────────────────────────

def _parse_file(base_name: str) -> Dict | None:
    """
    Parse one SROIE sample (box file + key JSON + image).
    Returns a dict with words, normalized bboxes, and NER tag ids; or None on error.
    """
    img_path = os.path.join(IMG_DIR, base_name + ".jpg")
    box_path = os.path.join(BOX_DIR, base_name + ".csv")
    key_path = os.path.join(KEY_DIR, base_name + ".json")

    if not all(os.path.exists(p) for p in [img_path, box_path, key_path]):
        return None

    # Image size for coordinate normalisation (LayoutLM uses 0–1000 range)
    try:
        with Image.open(img_path) as img:
            width, height = img.size
    except Exception:
        return None

    # Key JSON
    with open(key_path, "r", encoding="utf-8") as f:
        try:
            keys = json.load(f)
        except json.JSONDecodeError:
            return None

    words, boxes = [], []

    with open(box_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", 8)
            if len(parts) < 9:
                continue

            # SROIE quad format: x1,y1,x2,y2,x3,y3,x4,y4,text
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

            if x_min >= x_max or y_min >= y_max:
                continue

            words.append(text)
            boxes.append([x_min, y_min, x_max, y_max])

    if not words:
        return None

    ner_tags_str = _assign_ner_tags(words, keys)
    ner_tag_ids  = [LABEL2ID[t] for t in ner_tags_str]

    return {
        "id":       base_name,
        "words":    words,
        "bboxes":   boxes,
        "ner_tags": ner_tag_ids,
    }


# ─── Full dataset loading ────────────────────────────────────────────────────

def load_sroie_data() -> Tuple[List[Dict], List[Dict]]:
    """
    Load and split the SROIE dataset into train and val lists.
    Returns (train_examples, val_examples).
    """
    filenames = sorted([f for f in os.listdir(BOX_DIR) if f.endswith(".csv")])
    examples = []

    for fname in tqdm(filenames, desc="Parsing SROIE"):
        record = _parse_file(fname.replace(".csv", ""))
        if record is not None:
            examples.append(record)

    # Deterministic shuffle + split
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(examples)
    split = int(len(examples) * (1 - VAL_RATIO))
    return examples[:split], examples[split:]


# ─── Tokenization ────────────────────────────────────────────────────────────

_tokenizer: AutoTokenizer | None = None

def get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def tokenize_and_align_labels(examples: Dict) -> Dict:
    """
    Tokenize a batch of examples with WordPiece and propagate NER labels to
    every subword token.  Continuation subwords within the same word reuse the
    word-level label (not -100) so the model sees complete supervision.
    [CLS]/[SEP]/padding positions receive label -100 (ignored by loss).
    """
    tokenizer = get_tokenizer()

    tokenized = tokenizer(
        examples["words"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )

    all_labels = []
    all_bboxes = []

    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids       = tokenized.word_ids(batch_index=i)
        prev_word_idx  = None
        label_ids      = []
        bbox_list      = []

        for word_idx in word_ids:
            if word_idx is None:
                # [CLS], [SEP], or padding
                label_ids.append(-100)
                bbox_list.append([0, 0, 0, 0])
            else:
                label_ids.append(label_seq[word_idx])
                bbox_list.append(examples["bboxes"][i][word_idx])
            prev_word_idx = word_idx

        all_labels.append(label_ids)
        all_bboxes.append(bbox_list)

    tokenized["labels"] = all_labels
    tokenized["bbox"]   = all_bboxes
    return tokenized


def build_hf_datasets():
    """
    Build HuggingFace Dataset objects ready for the Trainer.
    Returns (train_dataset, val_dataset) with tokenized features.
    """
    train_raw, val_raw = load_sroie_data()

    features = Features({
        "id":       Value("string"),
        "words":    Sequence(Value("string")),
        "bboxes":   Sequence(Sequence(Value("int64"))),
        "ner_tags": Sequence(ClassLabel(names=LABELS)),
    })

    train_ds = Dataset.from_list(train_raw, features=features)
    val_ds   = Dataset.from_list(val_raw,   features=features)

    remove_cols = train_ds.column_names

    train_tokenized = train_ds.map(
        tokenize_and_align_labels, batched=True, remove_columns=remove_cols,
        desc="Tokenising train",
    )
    val_tokenized = val_ds.map(
        tokenize_and_align_labels, batched=True, remove_columns=remove_cols,
        desc="Tokenising val",
    )

    return train_tokenized, val_tokenized
