import os
import requests
import json

API_URL = "http://localhost:7860/predict"
MOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), "mockData")

def normalize_label(label):
    if not label:
        return ""
    # Remove spaces and convert to lowercase for comparison
    return label.replace(" ", "").replace("_", "").lower()

def run_tests():
    if not os.path.exists(MOCK_DATA_DIR):
        print(f"Error: mockData directory not found at {MOCK_DATA_DIR}")
        return

    images_tested = 0
    all_correct = 0
    results = []

    print(f"Starting test against {API_URL}...")
    
    # Get all subdirectories (labels)
    try:
        categories = [d for d in os.listdir(MOCK_DATA_DIR) if os.path.isdir(os.path.join(MOCK_DATA_DIR, d))]
    except Exception as e:
        print(f"Error listing directory: {e}")
        return
    
    for category in categories:
        folder_path = os.path.join(MOCK_DATA_DIR, category)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Testing {category}: {len(image_files)} images")
        
        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            
            try:
                with open(img_path, "rb") as f:
                    # Provide filename and content type explicitly to avoid 500 error on server
                    files = {"file": (img_name, f, "image/jpeg")}
                    response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    prediction = data.get("prediction", "")
                    confidence = data.get("confidence", 0)
                    
                    is_correct = normalize_label(prediction) == normalize_label(category)
                    
                    if is_correct:
                        all_correct += 1
                    
                    images_tested += 1
                    results.append({
                        "category": category,
                        "file": img_name,
                        "prediction": prediction,
                        "correct": is_correct,
                        "confidence": confidence
                    })
                else:
                    print(f"  [Error] {img_name}: {response.status_code}")
            except Exception as e:
                print(f"  [Failed] {img_name}: {str(e)}")

    if images_tested > 0:
        accuracy = all_correct / images_tested
        print("\n" + "="*40)
        print(f"Total Images: {images_tested}")
        print(f"Correct:      {all_correct}")
        print(f"Accuracy:     {accuracy:.2%}")
        print("="*40)
        
        # Optionally show some failures
        failures = [r for r in results if not r["correct"]]
        if failures:
            print(f"\nExample Failures (showing up to 5):")
            for f in failures[:5]:
                print(f"  Expected: {f['category']}, Got: '{f['prediction']}' (Confidence: {f['confidence']:.2f})")
    else:
        print("No images found to test.")

if __name__ == "__main__":
    run_tests()