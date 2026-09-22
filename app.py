import os
import subprocess

try:
    print("Attempting to pull Git LFS files...")
    subprocess.run(["git", "lfs", "pull"], check=True)
    print("Git LFS pull completed.")
except Exception as e:
    print(f"Git LFS pull skipped or failed: {e}")

import io
import torch
import numpy as np
from PIL import Image
import streamlit as st
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image

# Set page config before any other st command
st.set_page_config(page_title="Loan Document AI Prototype", layout="wide")

from autoencoder.config import DEVICE, OUTPUT_DIR, CHECKPOINT_DIR, DATA_DIR
from autoencoder.model import DocumentVAE
from transformers import AutoTokenizer, AutoModelForTokenClassification
from autoencoder.evaluate import load_best_model, compute_psnr
from autoencoder.losses import SSIMLoss
from autoencoder.dataset import ResizeWithPad

@st.cache_resource
def load_models():
    models = {}
    
    try:
        model, checkpoint = load_best_model()
        ssim_fn = SSIMLoss().to(DEVICE)
        models['ae'] = model
        models['ssim_fn'] = ssim_fn
    except Exception as e:
        st.warning(f"Failed to load Autoencoder model: {e}")
        
    try:
        vae_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_vae.pth")
        if not os.path.exists(vae_ckpt_path):
            vae_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_vae.pth")
        vae_model = DocumentVAE().to(DEVICE)
        vae_checkpoint = torch.load(vae_ckpt_path, map_location=DEVICE, weights_only=False)
        vae_model.load_state_dict(vae_checkpoint["model_state_dict"])
        vae_model.eval()
        models['vae'] = vae_model
    except Exception as e:
        st.warning(f"Failed to load VAE model: {e}")
        
    try:
        from transformer.config import TRANSFORMER_CKPT
        transformer_model_path = os.path.join(TRANSFORMER_CKPT, "best_model")
        if os.path.exists(transformer_model_path):
            transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_model_path)
            transformer_model = AutoModelForTokenClassification.from_pretrained(transformer_model_path)
            transformer_model.eval()
            models['transformer'] = transformer_model
            models['tokenizer'] = transformer_tokenizer
    except Exception as e:
        st.warning(f"Failed to load Transformer model: {e}")
        
    return models

models = load_models()
resize_pad = ResizeWithPad(512)
to_tensor = T.ToTensor()

st.title("Loan Document Processing using Generative AI")
st.markdown("This prototype demonstrates document processing and field extraction pipelines.")

# Sidebar navigation
page = st.sidebar.radio("Navigation", [
    "Upload & Overview", 
    "AE Reconstruction", 
    "VAE Scenario Generation", 
    "Transformer Field Extraction", 
    "Risk & Discrepancy Flags",
    "Real vs Synthetic Quality (Phase 2)",
    "Model Information"
])

# File uploader in sidebar so it persists
uploaded_file = st.sidebar.file_uploader("Upload Document (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    padded_img = resize_pad(image)
    input_tensor = to_tensor(padded_img).unsqueeze(0).to(DEVICE)
else:
    image = None
    padded_img = None
    input_tensor = None

if page == "Upload & Overview":
    st.header("Document Upload")
    if image is not None:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        st.info("Please upload an image from the sidebar to proceed.")

elif page == "AE Reconstruction":
    st.header("Autoencoder Reconstruction")
    if image is None:
        st.warning("Upload an image first.")
    else:
        if 'ae' in models and 'ssim_fn' in models:
            with st.spinner("Reconstructing..."):
                with torch.no_grad():
                    output_tensor = models['ae'](input_tensor)
                    ssim_val = models['ssim_fn'].compute_ssim_value(output_tensor, input_tensor)
                    psnr_val = compute_psnr(output_tensor, input_tensor)
                    reconstructed_img = to_pil_image(output_tensor.squeeze(0).cpu())
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(padded_img, caption="Original (Padded)", use_container_width=True)
            with col2:
                st.image(reconstructed_img, caption="Reconstructed", use_container_width=True)
                
            st.metric("SSIM", round(float(ssim_val), 4))
            st.metric("PSNR", round(float(psnr_val), 2))
        else:
            st.error("Autoencoder model not loaded.")

elif page == "VAE Scenario Generation":
    st.header("VAE Scenario Generation")
    if image is None:
        st.warning("Upload an image first.")
    else:
        if 'vae' in models:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.35, 0.05)
            if st.button("Generate Variations"):
                with st.spinner("Generating..."):
                    with torch.no_grad():
                        st.image(padded_img, caption="Original", width=300)
                        cols = st.columns(3)
                        for i in range(3):
                            output_tensor, _, _ = models['vae'](input_tensor, temperature=temperature, sample_mode=False)
                            sample_img = to_pil_image(output_tensor.squeeze(0).cpu())
                            with cols[i]:
                                st.image(sample_img, caption=f"Variation {i+1}", use_container_width=True)
        else:
            st.error("VAE model not loaded.")

elif page == "Transformer Field Extraction":
    st.header("Transformer Field Extraction")
    if image is None:
        st.warning("Upload an image first.")
    else:
        if 'transformer' in models and 'tokenizer' in models:
            with st.spinner("Extracting fields..."):
                base_name = os.path.splitext(uploaded_file.name)[0]
                box_path = os.path.join(os.path.dirname(DATA_DIR), "box", f"{base_name}.csv")
                
                if not os.path.exists(box_path):
                    st.error(f"OCR box data for {uploaded_file.name} not found in SROIE dataset.")
                else:
                    width, height = image.size
                    words, boxes = [], []
                    
                    with open(box_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split(",", 8)
                            if len(parts) < 9: continue
                            x1, y1, x2, y2, x3, y3, x4, y4 = map(int, parts[:8])
                            text = parts[8].strip()
                            if not text: continue
                            
                            x_min = max(0, min(1000, int((min(x1, x2, x3, x4) / width) * 1000)))
                            y_min = max(0, min(1000, int((min(y1, y2, y3, y4) / height) * 1000)))
                            x_max = max(0, min(1000, int((max(x1, x2, x3, x4) / width) * 1000)))
                            y_max = max(0, min(1000, int((max(y1, y2, y3, y4) / height) * 1000)))
                            
                            if x_min >= x_max or y_min >= y_max: continue
                            
                            words.append(text)
                            boxes.append([x_min, y_min, x_max, y_max])
                            
                    encoding = models['tokenizer'](
                        words, return_tensors="pt", truncation=True, 
                        is_split_into_words=True, padding="max_length", max_length=512
                    )
                    
                    word_ids = encoding.word_ids()
                    bbox_list = []
                    for word_idx in word_ids:
                        if word_idx is None:
                            bbox_list.append([0, 0, 0, 0])
                        else:
                            bbox_list.append(boxes[word_idx])
                            
                    encoding["bbox"] = torch.tensor([bbox_list])
                    
                    with torch.no_grad():
                        outputs = models['transformer'](**encoding)
                        
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predictions = outputs.logits.argmax(-1).squeeze().tolist()
                    confidences = probs.max(-1).values.squeeze().tolist()
                    
                    id2label = models['transformer'].config.id2label
                    
                    extracted = {"COMPANY": [], "ADDRESS": [], "DATE": [], "TOTAL": []}
                    extracted_conf = {"COMPANY": [], "ADDRESS": [], "DATE": [], "TOTAL": []}
                    
                    current_entity = None
                    current_words = []
                    current_confs = []
                    
                    previous_word_idx = None
                    for i, word_idx in enumerate(word_ids):
                        if word_idx is None or word_idx == previous_word_idx: continue
                        
                        label = id2label[predictions[i]]
                        word = words[word_idx]
                        conf = confidences[i]
                        
                        if label.startswith("B-"):
                            if current_entity:
                                extracted[current_entity].append(" ".join(current_words))
                                extracted_conf[current_entity].append(sum(current_confs)/len(current_confs))
                            current_entity = label[2:]
                            current_words = [word]
                            current_confs = [conf]
                        elif label.startswith("I-") and current_entity == label[2:]:
                            current_words.append(word)
                            current_confs.append(conf)
                        else:
                            if current_entity:
                                extracted[current_entity].append(" ".join(current_words))
                                extracted_conf[current_entity].append(sum(current_confs)/len(current_confs))
                                current_entity = None
                                current_words = []
                                current_confs = []
                        previous_word_idx = word_idx
                        
                    if current_entity:
                        extracted[current_entity].append(" ".join(current_words))
                        extracted_conf[current_entity].append(sum(current_confs)/len(current_confs))
                        
                    st.subheader("Extracted Fields")
                    for k, v in extracted.items():
                        if v:
                            val_str = " ".join(v).strip()
                            conf = round(sum(extracted_conf[k]) / len(extracted_conf[k]), 2)
                            
                            color = "green" if conf > 0.8 else "orange" if conf > 0.5 else "red"
                            st.markdown(f"**{k}**: {val_str} *(Confidence: <span style='color:{color}'>{conf}</span>)*", unsafe_allow_html=True)
                            
                    # Clean up output for risk rules
                    final_extraction = {}
                    for k, v in extracted.items():
                        if v:
                            final_extraction[k] = " ".join(v).strip()
                            
                    st.session_state['extracted_fields'] = final_extraction
                            
        else:
            st.error("Transformer model not loaded.")

elif page == "Risk & Discrepancy Flags":
    st.header("Risk & Discrepancy Flags")
    if image is None:
        st.warning("Upload an image first.")
    else:
        if 'extracted_fields' not in st.session_state:
            st.info("Run the Transformer Field Extraction first to populate this section.")
        else:
            final_extraction = st.session_state['extracted_fields']
            risks = []
            if not final_extraction.get("TOTAL"):
                risks.append({"field": "TOTAL", "issue": "Missing Total amount"})
            else:
                try:
                    # Try to parse float
                    total_val = float(final_extraction["TOTAL"].replace(",", "").replace("$", ""))
                    if total_val > 500:
                        risks.append({"field": "TOTAL", "issue": "High value amount detected (> 500)"})
                except:
                    risks.append({"field": "TOTAL", "issue": "Invalid format"})
                    
            if not final_extraction.get("DATE"):
                risks.append({"field": "DATE", "issue": "Missing Date"})
                
            if len(risks) == 0:
                st.success("No risks or discrepancies found in the extracted fields.")
            else:
                for risk in risks:
                    st.error(f"**Flag on {risk['field']}**: {risk['issue']}")

elif page == "Real vs Synthetic Quality (Phase 2)":
    st.header("Real vs Synthetic Quality")
    st.info("Coming in Phase 2")

elif page == "Model Information":
    st.header("Model Information")
    st.markdown("""
    ### 1. Convolutional Autoencoder (AE)
    - **Purpose**: Reconstructs the input document image with maximum visual and text fidelity. Acts as the cleanup and denoising stage before data extraction.
    - **Architecture**: Stacked Conv2D with progressively downsampling, bottleneck, and stacked ConvTranspose2D for upsampling.
    
    ### 2. Convolutional Variational Autoencoder (VAE)
    - **Purpose**: Learns a latent distribution over document appearance. Used to sample plausible document-layout variations for robustness testing.
    - **Architecture**: Similar to AE but with reparameterization trick (`mu`, `log_var`).
    
    ### 3. Document Transformer
    - **Purpose**: Reads the (AE-cleaned) document and extracts structured key fields (COMPANY, DATE, ADDRESS, TOTAL).
    - **Architecture**: LayoutLM-style token classification using text, 2D layout bounding-box embeddings, and image features.
    """)
    
    st.subheader("Model Loading Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("AE Model", "Loaded" if 'ae' in models else "Failed/Not Found")
    with col2:
        st.metric("VAE Model", "Loaded" if 'vae' in models else "Failed/Not Found")
    with col3:
        st.metric("Transformer", "Loaded" if 'transformer' in models else "Failed/Not Found")
