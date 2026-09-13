# Loan Document Processing using Generative AI
### Implementation Spec — Phase 1 (AE + VAE + Transformer)

## 1. Project Goal
Build a decision-support pipeline that takes a scanned loan-related document image, cleans/reconstructs it, understands its structure, and extracts key fields for downstream risk-flagging. This is a **research/college prototype**, not an autonomous loan-approval system.

**Use case (source):** "Loan Document Processing using Gen AI" — extract essential information from large volumes of unstructured documents (pay stubs, tax returns, etc.), speed up processing, and identify discrepancies/inconsistencies.

## 2. Scope for This Phase
Implement and validate, in order:
1. Convolutional Autoencoder (AE)
2. Convolutional VAE
3. Document Transformer (field extraction)

**Explicitly out of scope for this phase** (build later, after the above are validated): Conditional GAN, Diffusion model. Stub these as empty modules/interfaces but do not implement logic yet.

## 3. Dataset
**Primary: SROIE (ICDAR 2019 Scanned Receipts OCR & Information Extraction)**
- 1,000 scanned receipt images, split 600 (trainval) / 400 (test)
- Two annotation types per image:
  - Text localization + transcript file: `x1,y1,x2,y2,x3,y3,x4,y4,transcript` per line
  - Key information JSON: `{"company": ..., "address": ..., "date": ..., "total": ...}`
- Source: linked from https://github.com/zzzDavid/ICDAR-2019-SROIE (README points to the actual Google Drive download; the repo itself is mostly code, not the raw dataset)
- License: MIT

**Secondary (already available, optional for transformer cross-training): FUNSD**
- 149 train / 50 test noisy scanned form images with word-level boxes, `label` (question/answer/header/other), and `linking` (key-value pairs)

Treat `company`→borrower/party name, `date`→document date, `total`→loan/payment amount, `address`→borrower address as the field-mapping analogy to loan documents in all reporting/writeups.

### Expected folder structure after setup
```
data/
  sroie/
    trainval/
      images/*.jpg
      boxes_transcripts/*.txt
      entities/*.json
    test/
      images/*.jpg
      boxes_transcripts/*.txt
      entities/*.json
  funsd/
    training_data/{images,annotations}/
    testing_data/{images,annotations}/
```

## 4. Model 1 — Convolutional Autoencoder (AE)
**Purpose:** Reconstruct the input document image with maximum visual/text fidelity; acts as the cleanup/denoising stage before extraction.

**Architecture**
- Encoder: stacked Conv2D + BatchNorm + LeakyReLU, progressively downsampling (e.g., 4–5 conv blocks)
- Bottleneck: dense or conv latent representation
- Decoder: stacked ConvTranspose2D (or upsample + conv) mirroring the encoder
- Input: resized/normalized document image (fix a consistent size, e.g., 512×512 or 768×768, preserve aspect ratio with padding)

**Loss function (composite, not plain MSE):**
```
L_total = λ1 * L_pixel_SSIM + λ2 * L_perceptual + λ3 * L_edge_text
```
- `L_pixel_SSIM`: 1 − SSIM(original, reconstructed), optionally combined with L1
- `L_perceptual`: feature distance from a pretrained VGG (frozen weights) between original and reconstructed
- `L_edge_text`: Sobel/gradient-based edge loss, or a pixel-weighted loss that upweights text-dense regions (use a simple text-mask heuristic from the OCR box annotations if available)
- Suggested starting weights: λ1=1.0, λ2=0.5, λ3=0.5 — tune empirically

**Evaluation metrics**
- SSIM and PSNR between original and reconstructed
- OCR accuracy delta: run Tesseract (or the dataset's ground-truth transcripts) on original vs. reconstructed, compare extracted text accuracy — this is the metric that actually matters for this use case, not just visual SSIM

**Output artifact:** trained encoder+decoder checkpoint; encoder reused as feature extractor if needed later.

## 5. Model 2 — Convolutional VAE
**Purpose:** Learn a latent distribution over document appearance; sample from it to generate plausible document-layout variations for robustness testing of the extraction pipeline (not for producing final "real" synthetic documents — that's the GAN/diffusion job later).

**Architecture**
- Same conv encoder/decoder skeleton as the AE, but encoder outputs `mu` and `log_var` instead of a deterministic latent
- Reparameterization trick for sampling
- Decoder identical structure to AE decoder

**Loss function**
```
L_VAE = L_reconstruction + β * KL_divergence(q(z|x) || N(0,1))
```
- Reconstruction term can reuse the AE's SSIM+perceptual combo (lighter weight is fine here)
- Use β-VAE weighting (start β low, e.g. 0.001–0.01, and tune — small dataset means KL term can easily dominate and collapse the latent space)

**Evaluation**
- Visual inspection grid of sampled reconstructions at varying latent temperature
- Latent space traversal plot (interpolate between two real document encodings, confirm smooth/plausible transitions)
- Do not claim high-fidelity generative realism given dataset size (~600–1000 images) — this stage is for exploration/robustness testing, state that clearly in the report

## 6. Model 3 — Document Transformer
**Purpose:** Read the (AE-cleaned) document and extract structured key fields; perform basic cross-field consistency checks that stand in for "risk flagging."

**Approach**
- Use a LayoutLM-style architecture (text + 2D layout/bounding-box embeddings + image features), since SROIE/FUNSD provide exactly this input shape (OCR text + box coordinates)
- Recommended: fine-tune a pretrained layout-aware transformer (e.g., LayoutLMv3-base) rather than training from scratch — dataset size (600–1000 images) is too small for from-scratch transformer training
- Task framing: token/entity classification — label each OCR token as one of `{company/party, date, address, total/amount, other}` (SROIE) or `{question, answer, header, other}` (FUNSD)

**Discrepancy/"risk" layer (rule-based, on top of extracted fields — not learned)**
- Example checks to implement: missing required field, date format/range sanity, amount field non-numeric or zero, mismatched party name across linked documents (if multi-doc)
- Document clearly in the report/README that this layer is deterministic rule logic applied to transformer output, not a learned risk model — there is no risk-labeled training data in SROIE/FUNSD

**Evaluation**
- Entity-level precision/recall/F1 per field
- Compare against SROIE Task 3 baseline (Bi-LSTM, 75.58% Hmean) as a reference point

## 7. Pipeline Wiring
```
Document Image
     ↓
Conv-AE  →  Cleaned Reconstruction
     ↓
Conv-VAE (branch, robustness/scenario sampling — optional per-run)
     ↓
Document Transformer (on cleaned reconstruction)
     ↓
Field Extraction (JSON output: company/party, date, address, amount)
     ↓
Rule-based Discrepancy/Risk Flags
     ↓
Streamlit UI
```

## 8. Streamlit App Requirements
Pages/sections:
1. **Upload** — accept a document image (jpg/png)
2. **AE Reconstruction** — side-by-side original vs. reconstructed, plus SSIM/PSNR score
3. **VAE Scenario Generation** — show 3–5 sampled latent variations of the uploaded doc
4. **Transformer Field Extraction** — display extracted fields as a table, highlight low-confidence extractions
5. **Risk/Discrepancy Flags** — list any rule-based flags triggered, with plain-language explanation
6. **Real vs Synthetic Quality Comparison** — placeholder page for Phase 2 (GAN/diffusion), can be stubbed with "Coming in Phase 2" for now

Keep Streamlit stateless per-session (no DB required for the prototype); cache model loading with `@st.cache_resource`.

## 9. Tech Stack
- PyTorch (model implementation)
- HuggingFace Transformers (`layoutlmv3-base` or similar, for Model 3)
- OpenCV / Pillow (image preprocessing)
- scikit-image (SSIM/PSNR)
- pytesseract (OCR accuracy evaluation, optional if ground-truth transcripts are used instead)
- Streamlit (UI)

## 10. Milestones / Order of Implementation
1. Data loading + preprocessing pipeline for SROIE (images + boxes + entity JSON → normalized tensors)
2. Conv-AE: train, validate with SSIM/PSNR/OCR-delta, save checkpoint
3. Conv-VAE: train, validate with latent traversal + sample quality, save checkpoint
4. Document Transformer: fine-tune on SROIE (+ optionally FUNSD), validate with entity F1
5. Rule-based discrepancy/risk layer on top of Transformer output
6. Streamlit app wiring all of the above
7. **(Phase 2, not now)** cGAN for synthetic document generation
8. **(Phase 2, not now)** Diffusion model, likely replacing/supplementing the GAN

## 11. Explicit Non-Goals for This Phase
- Do not implement GAN or diffusion model code yet — stub only
- Do not present the risk-flagging output as a validated credit-risk model — it is rule-based logic for a research prototype
- Do not use real borrower PII — SROIE/FUNSD data only
