import torch
import torch.nn as nn
from torchvision import transforms
from src.model import create_model, CLASS_NAMES
from PIL import Image
import io
import os
import mlflow.pytorch
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation for input image
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Global variables for model
model = None
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Thai Food Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_model():
    global model
    # Try to load the model from MLflow Model Registry first (Production stage)
    # If not available, look for a local backup or wait for the first DAG to run
    model_name = "googlenet-thai-food"
    try:
        model_uri = f"models:/{model_name}/Production"
        print(f"Loading model from {model_uri}...")
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model.eval()
        print(f"Model loaded successfully from MLflow on {device}")
    except Exception as e:
        print(f"Could not load model from MLflow: {e}")
        traceback.print_exc()
        # Fallback to local file if available (for initial startup test)
        local_path = "best_googlenet_thai_food.pth"
        if os.path.exists(local_path):
            try:
                checkpoint = torch.load(local_path, map_location=device)
                num_classes = checkpoint.get("num_classes", len(CLASS_NAMES))
                model = create_model(num_classes)
                # Using weights_only=False carefully here because we know the source, 
                # but ideally we should transition to MLflow completely.
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                model.to(device)
                model.eval()
                print(f"Model loaded successfully from local file on {device}")
            except Exception as le:
                print(f"Error loading local model: {le}")

@app.post("/reload")
def reload_model_endpoint():
    try:
        load_model()
        return {"message": "Model reloaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail={"status": "degraded", "model_loaded": False})
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        # Read image
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Transform
        x = val_transform(img)
        x = x.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        
        threshold = 0.52
        is_confident = confidence >= threshold
        
        return {
            "prediction": label if is_confident else "",
            "confidence": confidence,
            "class_id": pred_idx if is_confident else -1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
