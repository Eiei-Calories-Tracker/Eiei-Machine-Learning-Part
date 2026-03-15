import argparse
import os
import torch
from src.drift_utils import check_drift
from src.data_utils import get_latest_version

def run_drift_check_task(base_data_dir, mock_data_dir, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    latest_v = get_latest_version(base_data_dir)
    if not latest_v:
        print("No latest version found.")
        is_drift = True # Force drift if no version exists
    else:
        ref_dir = os.path.join(base_data_dir, latest_v, 'train')
        ref_images = []
        for root, _, files in os.walk(ref_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    ref_images.append(os.path.join(root, f))
        
        test_images = []
        for root, _, files in os.walk(mock_data_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_images.append(os.path.join(root, f))
        
        # Take a subset for speed if needed
        is_drift, p_val = check_drift(ref_images[:100], test_images[:100], device)
        print(f"Drift check result: is_drift={is_drift}, p_val={p_val}")

    with open("drift_result.txt", "w") as f:
        f.write("drift" if is_drift else "no_drift")
    
    return is_drift

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_dir", type=str, required=True)
    parser.add_argument("--mock_data_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    run_drift_check_task(
        base_data_dir=args.base_data_dir,
        mock_data_dir=args.mock_data_dir
    )

if __name__ == "__main__":
    main()
