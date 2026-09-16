"""
Generates transformer_train.ipynb from scratch.
Run: python make_transformer_nb.py
"""
import json, os

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [source]})

def code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    })

# ── Cells ──────────────────────────────────────────────────────────────────────
md("""# 📄 Document Transformer — LayoutLM Fine-Tuning on SROIE

Trains a **LayoutLM** model for token-level NER on the **ICDAR-2019 SROIE** receipt dataset.

**Fields extracted:** `COMPANY`, `DATE`, `ADDRESS`, `TOTAL`

**Pipeline:**
1. Environment setup
2. Load & parse SROIE data
3. Tokenise and align BIO labels
4. Fine-tune LayoutLM (30 epochs, early stopping)
5. Evaluate with seqeval (precision / recall / F1)
6. Save best checkpoint""")

md("## 0 — Environment Setup")

code(
"# Uncomment to install if needed\n"
"# !pip install transformers datasets seqeval evaluate Pillow tqdm torch"
)

code(
"import sys, os\n\n"
"PROJECT_ROOT = os.path.abspath('.')\n"
"if PROJECT_ROOT not in sys.path:\n"
"    sys.path.insert(0, PROJECT_ROOT)\n\n"
"print(f'Project root: {PROJECT_ROOT}')\n"
"print(f'Python:       {sys.version}')"
)

md("## 1 — Configuration")

code(
"from transformer.config import (\n"
"    BATCH_SIZE, DEVICE, EARLY_STOP_PATIENCE, EPOCHS, EVAL_STRATEGY,\n"
"    LABELS, LABEL2ID, ID2LABEL, LEARNING_RATE, LOGGING_STEPS, MAX_SEQ_LENGTH,\n"
"    METRIC_FOR_BEST, MODEL_NAME, NUM_LABELS, OUTPUT_DIR, RANDOM_SEED,\n"
"    SAVE_TOTAL_LIMIT, TRANSFORMER_CKPT, WARMUP_RATIO, WEIGHT_DECAY,\n"
"    DATA_DIR, BOX_DIR, KEY_DIR, IMG_DIR,\n"
")\n\n"
"print('=' * 60)\n"
"print('  Document Transformer — Configuration Summary')\n"
"print('=' * 60)\n"
"print(f'  Device          : {DEVICE}')\n"
"print(f'  Base model      : {MODEL_NAME}')\n"
"print(f'  Epochs          : {EPOCHS}')\n"
"print(f'  Batch size      : {BATCH_SIZE}')\n"
"print(f'  Learning rate   : {LEARNING_RATE}')\n"
"print(f'  Max seq length  : {MAX_SEQ_LENGTH}')\n"
"print(f'  Num labels      : {NUM_LABELS}')\n"
"print(f'  Labels          : {LABELS}')\n"
"print(f'  Checkpoint dir  : {TRANSFORMER_CKPT}')\n"
"print('=' * 60)"
)

md("## 2 — Load & Parse SROIE Dataset")

code(
"from transformer.dataset import load_sroie_data\n\n"
"print('[1/4] Loading and parsing SROIE dataset ...')\n"
"train_raw, val_raw = load_sroie_data()\n\n"
"print(f'  Train samples : {len(train_raw)}')\n"
"print(f'  Val   samples : {len(val_raw)}')\n\n"
"sample = train_raw[0]\n"
"print(f\"--- Sample preview (id={sample['id']}) ---\")\n"
"print(f\"  Words  (first 10): {sample['words'][:10]}\")\n"
"print(f\"  Boxes  (first 10): {sample['bboxes'][:10]}\")\n"
"print(f\"  Tags   (first 10): {[LABELS[t] for t in sample['ner_tags'][:10]]}\")"
)

md("## 3 — Tokenise & Build HuggingFace Datasets")

code(
"from transformer.dataset import build_hf_datasets\n\n"
"print('Tokenising datasets (this may take a minute) ...')\n"
"train_ds, val_ds = build_hf_datasets()\n\n"
"print(f'  Train dataset : {train_ds}')\n"
"print(f'  Val   dataset : {val_ds}')"
)

md("## 4 — Initialise LayoutLM Model")

code(
"from transformer.model import DocumentTransformer\n"
"from transformer.dataset import get_tokenizer\n\n"
"print('[2/4] Initialising LayoutLM model ...')\n"
"doc_transformer = DocumentTransformer.from_pretrained()\n"
"model     = doc_transformer.model\n"
"tokenizer = get_tokenizer()\n\n"
"print(f'  Model type    : {type(model).__name__}')\n"
"print(f'  Num labels    : {model.config.num_labels}')\n"
"print(f'  id2label      : {model.config.id2label}')"
)

md("## 5 — Configure Trainer")

code(
"from transformers import (\n"
"    DataCollatorForTokenClassification,\n"
"    EarlyStoppingCallback,\n"
"    TrainingArguments,\n"
"    Trainer,\n"
")\n"
"from transformer.evaluate import compute_metrics\n\n"
"print('[3/4] Configuring Trainer ...')\n\n"
"training_args = TrainingArguments(\n"
"    output_dir                  = TRANSFORMER_CKPT,\n"
"    eval_strategy               = EVAL_STRATEGY,\n"
"    save_strategy               = EVAL_STRATEGY,\n"
"    learning_rate               = LEARNING_RATE,\n"
"    per_device_train_batch_size = BATCH_SIZE,\n"
"    per_device_eval_batch_size  = BATCH_SIZE,\n"
"    num_train_epochs            = EPOCHS,\n"
"    weight_decay                = WEIGHT_DECAY,\n"
"    warmup_steps                = int(WARMUP_RATIO * EPOCHS * 100),\n"
"    logging_steps               = LOGGING_STEPS,\n"
"    save_total_limit            = SAVE_TOTAL_LIMIT,\n"
"    load_best_model_at_end      = True,\n"
"    metric_for_best_model       = METRIC_FOR_BEST,\n"
"    greater_is_better           = True,\n"
"    seed                        = RANDOM_SEED,\n"
"    report_to                   = 'none',\n"
"    fp16                        = True,  # Enables mixed precision to save VRAM and avoid cuBLAS errors\n"
")\n\n"
"data_collator = DataCollatorForTokenClassification(tokenizer)\n\n"
"trainer = Trainer(\n"
"    model            = model,\n"
"    args             = training_args,\n"
"    train_dataset    = train_ds,\n"
"    eval_dataset     = val_ds,\n"
"    processing_class = tokenizer,\n"
"    data_collator    = data_collator,\n"
"    compute_metrics  = compute_metrics,\n"
"    callbacks        = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],\n"
")\n\n"
"print(f'  Epochs              : {EPOCHS}')\n"
"print(f'  Steps/epoch (est.)  : {len(train_ds) // BATCH_SIZE}')\n"
"print(f'  Early stop patience : {EARLY_STOP_PATIENCE} epochs')\n"
"print('  Trainer ready ✓')"
)

md("## 6 — Train")

code(
"print('[4/4] Starting training ...')\n"
"train_result = trainer.train()\n\n"
"print('\\n' + '=' * 60)\n"
"print('  Training complete!')\n"
"print(f'  Total steps    : {train_result.global_step}')\n"
"print(f'  Train loss     : {train_result.training_loss:.4f}')\n"
"print('=' * 60)"
)

md("## 7 — Save Best Model")

code(
"import os\n"
"best_model_dir = os.path.join(TRANSFORMER_CKPT, 'best_model')\n"
"trainer.save_model(best_model_dir)\n"
"tokenizer.save_pretrained(best_model_dir)\n"
"print(f'  ✓  Best model saved → {best_model_dir}')"
)

md("## 8 — Final Evaluation")

code(
"from transformer.evaluate import evaluate_checkpoint\n\n"
"metrics = evaluate_checkpoint(trainer, split='val')\n\n"
"print('--- Metrics Summary ---')\n"
"for k, v in sorted(metrics.items()):\n"
"    print(f'  {k:35s}: {v}')"
)

md("## 9 — Save Training Summary JSON")

code(
"import json, os\n\n"
"summary = {\n"
"    'global_step':   train_result.global_step,\n"
"    'training_loss': train_result.training_loss,\n"
"    'final_metrics': metrics,\n"
"}\n\n"
"summary_path = os.path.join(OUTPUT_DIR, 'transformer_training_summary.json')\n"
"with open(summary_path, 'w') as f:\n"
"    json.dump(summary, f, indent=2)\n\n"
"print(f'  ✓  Training summary saved → {summary_path}')\n"
"print('  Done! 🎉')"
)

md("---\n## 10 — (Optional) Quick Inference Test")

code(
"from transformer.model import DocumentTransformer\n"
"import torch\n\n"
"loaded = DocumentTransformer.load_checkpoint()\n\n"
"sample = val_raw[0]\n"
"words  = sample['words']\n"
"boxes  = sample['bboxes']\n\n"
"enc = tokenizer(\n"
"    words,\n"
"    is_split_into_words=True,\n"
"    truncation=True,\n"
"    padding='max_length',\n"
"    max_length=MAX_SEQ_LENGTH,\n"
"    return_tensors='pt',\n"
")\n\n"
"word_ids    = enc.word_ids()\n"
"bbox_tensor = torch.tensor(\n"
"    [boxes[w] if w is not None else [0, 0, 0, 0] for w in word_ids]\n"
").unsqueeze(0)\n\n"
"tokenized_input = {\n"
"    'input_ids':      enc['input_ids'],\n"
"    'attention_mask': enc['attention_mask'],\n"
"    'token_type_ids': enc['token_type_ids'],\n"
"    'bbox':           bbox_tensor,\n"
"}\n\n"
"extracted = loaded.extract_fields(tokenized_input, words, word_ids)\n\n"
"print(f\"Sample ID: {sample['id']}\")\n"
"print('Extracted fields:')\n"
"for field, value in extracted.items():\n"
"    print(f\"  {field:10s}: {value if value else '(not detected)'}\")"
)

# ── Build notebook dict ────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(__file__), "transformer_train.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out}")
