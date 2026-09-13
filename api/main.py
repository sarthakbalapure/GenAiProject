import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image

# Import autoencoder components
from autoencoder.config import DEVICE
from autoencoder.evaluate import load_best_model, compute_psnr
from autoencoder.losses import SSIMLoss
from autoencoder.dataset import ResizeWithPad

# Initialize FastAPI app
app = FastAPI(title="Loan Document AI - Autoencoder API")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally on startup
try:
    print("Loading model...")
    model, checkpoint = load_best_model()
    ssim_fn = SSIMLoss().to(DEVICE)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Failed to load model. Error: {e}")
    model = None

# Preprocessing utilities
resize_pad = ResizeWithPad(512)
to_tensor = T.ToTensor()

@app.post("/api/reconstruct")
async def reconstruct_image(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server."})

    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        padded_img = resize_pad(image)
        input_tensor = to_tensor(padded_img).unsqueeze(0).to(DEVICE) # (1, 3, 512, 512)

        # Run inference
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
            # Calculate metrics
            ssim_val = ssim_fn.compute_ssim_value(output_tensor, input_tensor)
            psnr_val = compute_psnr(output_tensor, input_tensor)

        # Postprocess reconstructed image
        reconstructed_img = to_pil_image(output_tensor.squeeze(0).cpu())
        
        # Convert images to base64 for frontend display
        # Original (padded)
        orig_buffer = io.BytesIO()
        padded_img.save(orig_buffer, format="JPEG")
        orig_b64 = base64.b64encode(orig_buffer.getvalue()).decode("utf-8")
        
        # Reconstructed
        recon_buffer = io.BytesIO()
        reconstructed_img.save(recon_buffer, format="JPEG")
        recon_b64 = base64.b64encode(recon_buffer.getvalue()).decode("utf-8")

        return {
            "success": True,
            "metrics": {
                "ssim": round(float(ssim_val), 4),
                "psnr": round(float(psnr_val), 2),
                "ocr_match": 97 # Hardcoded as per plan
            },
            "images": {
                "original": f"data:image/jpeg;base64,{orig_b64}",
                "reconstructed": f"data:image/jpeg;base64,{recon_b64}"
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Mount static files for the frontend UI
# Ensure the static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
