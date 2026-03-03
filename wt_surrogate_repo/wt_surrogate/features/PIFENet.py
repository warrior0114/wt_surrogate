# --- PIFENet: Enhanced Feature Extractor ---

import os
import torch
import torch.nn as nn
import numpy as np

# --- Test Function ---
def test_v8_extractor():
    """Tests the V8 version of the feature extractor."""
    WEIGHTS_DIR = 'precomputed_weights_v3'
    if not os.path.exists(WEIGHTS_DIR):
        print(f"Error: Weights directory '{WEIGHTS_DIR}' not found. Please run the weight generation script first.")
        # Create dummy weight files for testing convenience if directory does not exist
        print("Creating dummy weight files for testing...")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        weight_files = {
            'w_hub.npy': np.ones((31, 31)), 'w_yaw_gaussian_60deg.npy': np.ones((31, 31)),
            'w_blade_tip.npy': np.ones((31, 31)), 'w_corr_multi_target.npy': np.ones((31, 31)),
            'w_top.npy': np.ones((31, 31)), 'w_bottom.npy': np.ones((31, 31)),
            'w_left.npy': np.ones((31, 31)), 'w_right.npy': np.ones((31, 31))
        }
        for filename, arr in weight_files.items():
            np.save(os.path.join(WEIGHTS_DIR, filename), arr)
        print("Dummy weight files created.")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = PIFENet(weights_dir=WEIGHTS_DIR).to(device)
        model.eval()
        wind_matrix = torch.randn(8, 3, 31, 31).to(device)
        
        with torch.no_grad():
            features = model(wind_matrix)
        
        print(f"\n--- PIFENet Test ---")
        print(f"Input shape: {wind_matrix.shape}")
        print(f"Output feature shape: {features.shape}")
        
        print(f"\nFeature List ({len(model.feature_names)} features):")
        print("=" * 80)
        
        # Display features by group
        groups = [
            ("Global Basic Features", 0, 10),
            ("Weighted Regional Features", 10, 28),
            ("Turbulence Characteristics", 28, 34),
            ("Spatial Structure Features", 34, 39),
            ("Extreme Value Statistics", 39, 42),
            ("Energy Distribution Features", 42, 45)
        ]
        
        for group_name, start, end in groups:
            print(f"\n{group_name} ({end-start} features):")
            for i in range(start, end):
                if i < len(model.feature_names):
                    is_finite = torch.all(torch.isfinite(features[:, i])).item()
                    status = "✓ OK" if is_finite else "✗ NaN/Inf"
                    print(f"  {i+1:02d}: {model.feature_names[i]:<35} {status}")
        
        print("\n" + "=" * 80)
        print("Feature Statistics:")
        num_valid_features = torch.sum(torch.all(torch.isfinite(features), dim=0)).item()
        print(f"- Total Features: {len(model.feature_names)}")
        print(f"- Valid Features (no NaN/Inf): {num_valid_features}")
        if num_valid_features > 0:
            valid_features = features[:, torch.all(torch.isfinite(features), dim=0)]
            print(f"- Feature Range: [{valid_features.min().item():.4f}, {valid_features.max().item():.4f}]")
            print(f"- Feature Mean: {valid_features.mean().item():.4f}")
            print(f"- Feature Std Dev: {valid_features.std().item():.4f}")
        
        if num_valid_features < len(model.feature_names):
            print("\nWarning: Some feature calculations resulted in NaN or Inf. Please check the implementation.")

    except Exception as e:
        print(f"Error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_v8_extractor()