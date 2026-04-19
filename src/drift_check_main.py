import argparse
import os
import torch
import mlflow
from src.drift_utils import check_drift
from src.data_utils import get_latest_version
from src.mlflow_metadata import init_mlflow, safe_set_experiment, set_seed

def run_drift_check_task(base_data_dir, mock_data_dir, model_uri=None, **kwargs):
    set_seed(42)  # Ensure reproducibility for MMD permutation tests
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
        results = check_drift(ref_images[:100], test_images[:100], device, model_uri=model_uri)
        is_drift = results["is_drift"]
        p_val = results["p_val"]
        distance = results["distance"]
        threshold = results["threshold"]
        
        print(f"Drift check result: is_drift={is_drift}")
        print(f"  > p_val:     {p_val:.10f}")
        print(f"  > distance:  {distance:.10f}")
        print(f"  > threshold: {threshold:.10f}")

        # Log to MLflow for specialized monitoring
        init_mlflow()
        from src.mlflow_metadata import safe_set_experiment
        safe_set_experiment("ThaiFood_Monitoring")
        with mlflow.start_run(run_name="drift_check"):
            mlflow.log_param("model_uri", model_uri or "default_googlenet")
            mlflow.log_metrics({
                "is_drift": float(is_drift),
                "p_val": p_val,
                "distance": distance,
                "threshold": threshold
            })
            mlflow.set_tag("latest_version", latest_v)
            print("Successfully logged drift metrics to MLflow.")

    return is_drift

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_dir", type=str, required=True)
    parser.add_argument("--mock_data_dir", type=str, required=True)
    parser.add_argument("--model_uri", type=str, default=None)
    
    args = parser.parse_args()
    
    run_drift_check_task(
        base_data_dir=args.base_data_dir,
        mock_data_dir=args.mock_data_dir,
        model_uri=args.model_uri
    )

if __name__ == "__main__":
    main()
