import torch
import numpy as np
from alibi_detect.cd import MMDDrift
from PIL import Image
from src.model import create_model
from src.data_utils import get_transforms
import os

def get_features(model, image_paths, transform, device):
    model.eval()
    features = []
    with torch.no_grad():
        for p in image_paths:
            if not os.path.exists(p): continue
            img = Image.open(p).convert("RGB")
            # We need the features before the last linear layer
            # For GoogLeNet, we can temporarily set fc to Identity
            feat = model(transform(img).unsqueeze(0).to(device)).squeeze(0)
            features.append(feat.cpu().numpy())
    return np.array(features)

def check_drift(reference_images, test_images, device, p_val=0.05):
    """
    Checks for data drift between reference and test image sets using MMD.
    """
    _, val_tf = get_transforms()
    
    # Load model and strip the final head to get features
    model = create_model()
    # If we have a local base model, we should load it here. 
    # For now, we use the pre-trained features.
    model.fc = torch.nn.Identity()
    model.to(device)
    
    X_ref = get_features(model, reference_images, val_tf, device)
    X_test = get_features(model, test_images, val_tf, device)
    
    if len(X_ref) == 0 or len(X_test) == 0:
        return False, 1.0
        
    drift_detector = MMDDrift(
        x_ref=X_ref,
        backend="pytorch",
        p_val=p_val
    )
    
    prediction = drift_detector.predict(X_test)
    is_drift = int(prediction['data']['is_drift']) == 1
    p_value = prediction['data']['p_val']
    
    return is_drift, p_value
