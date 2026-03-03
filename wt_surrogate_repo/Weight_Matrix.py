import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
import struct
import numpy as np
import pprint
# Try to import the specific reading function


# ==============================================================================
# 1. Configuration
# ==============================================================================
GRID_SIZE = 31
PLANE_SIZE_M = 145.0
ROTOR_RADIUS_M = 61.5
YAW_ANGLE_DEG = 45.0 
BTS_DIR = 'Wind_bts'
TURBINE_DATA_DIR = 'load'
NUM_CASES_FOR_CORR = 50
TARGET_COLS_FOR_CORR = ["RootFxb1", "RootMyb1", "TwrBsFxt", "TwrBsMyt", "TTDspFA", "GenPwr"]
OUTPUT_DIR = 'weights'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_bts_official(filepath):
    """
    Reads and parses TurbSim .bts files.
    This version strictly follows the logic of the official 'turbsim_file.py' provided by the user.
    """
    try:
        with open(filepath, 'rb') as f:
            # --- 1. Read File Header (based on official code) ---
            header = {}
            
            # Read ID, dimensions, and time steps (2 + 4*4 = 18 bytes)
            (header['ID'], 
             header['nz'], 
             header['ny'], 
             header['nTwr'], 
             header['nt']) = struct.unpack('<h4l', f.read(18))

            # Read spacing and reference values (6*4 = 24 bytes)
            (header['dz'], 
             header['dy'], 
             header['dt'], 
             header['uHub'], 
             header['zHub'], 
             header['zBottom']) = struct.unpack('<6f', f.read(24))

            # Read scaling and offset factors for U, V, W (6*4 = 24 bytes)
            scl_off = struct.unpack('<6f', f.read(24))
            scl = np.float32([scl_off[0], scl_off[2], scl_off[4]])
            off = np.float32([scl_off[1], scl_off[3], scl_off[5]])
            header['scale_factors'] = scl
            header['offsets'] = off
            
            # Read length of the description string (4-byte integer)
            (nChar,) = struct.unpack('<l', f.read(4))
            
            # Read the description string itself
            header['info'] = f.read(nChar).decode('utf-8', errors='ignore').strip()

            # --- 2. Read Wind Speed Data ---
            # According to official code, it reads by time step in a loop
            # This correctly handles interlaced grid and tower data
            
            ny, nz, nt, nTwr = header['ny'], header['nz'], header['nt'], header['nTwr']
            
            # Initialize empty numpy arrays to store data
            u = np.zeros((3, nt, ny, nz), dtype=np.float32)
            uTwr = np.zeros((3, nt, nTwr), dtype=np.float32)

            for it in range(nt):
                # Read main grid data
                # 2 bytes (int16) * 3 components * ny * nz
                grid_bytes = 2 * 3 * ny * nz
                if grid_bytes > 0:
                    buffer_grid = np.frombuffer(f.read(grid_bytes), dtype=np.int16)
                    # Reshape using Fortran-style (order='F'), which is crucial
                    reshaped_grid = buffer_grid.astype(np.float32).reshape((3, ny, nz), order='F')
                    u[:, it, :, :] = reshaped_grid
                
                # Read tower data
                # 2 bytes (int16) * 3 components * nTwr
                tower_bytes = 2 * 3 * nTwr
                if tower_bytes > 0:
                    buffer_tower = np.frombuffer(f.read(tower_bytes), dtype=np.int16)
                    reshaped_tower = buffer_tower.astype(np.float32).reshape((3, nTwr), order='F')
                    uTwr[:, it, :] = reshaped_tower

            # --- 3. Apply Scaling and Offsets (based on official code) ---
            # Formula: (raw_value - offset) / scale
            # Use broadcasting for efficient calculation
            u -= off[:, None, None, None]
            u /= scl[:, None, None, None]

            if nTwr > 0:
                uTwr -= off[:, None, None]
                uTwr /= scl[:, None, None]
            
            # For ease of use, rearrange data axis order to (nt, 3, ny, nz)
            wind_data = np.transpose(u, (1, 0, 2, 3))
            
            return header, wind_data

    except Exception as e:
        print(f"Error occurred while reading file: {e}")
        return None, None

# Helper function for normalization
def _normalize(weights: np.ndarray) -> np.ndarray:
    s = weights.sum()
    return weights / s if s > 0 else weights

# Grid parameter calculations
center_idx = (GRID_SIZE - 1) / 2.0
m_per_cell = PLANE_SIZE_M / (GRID_SIZE - 1) if GRID_SIZE > 1 else 0
radius_in_cells = ROTOR_RADIUS_M / m_per_cell

print(f"Grid Parameters: Center Index={center_idx}, m/cell={m_per_cell:.2f}m, Rotor Radius={radius_in_cells:.2f} cells")

# ==============================================================================
# 3. Weight Matrix Generation Functions
# ==============================================================================
def generate_hub_weights() -> np.ndarray:
    print("Generating: w_hub (Gaussian weights)...")
    weights = np.zeros((GRID_SIZE, GRID_SIZE))
    sigma = center_idx / 2.0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            dist_sq = (i - center_idx)**2 + (j - center_idx)**2
            weights[i, j] = math.exp(-dist_sq / (2 * sigma**2))
    return _normalize(weights)

def generate_yaw_gaussian_weights(yaw_angle_deg: float) -> np.ndarray:
    """Generates elliptical Gaussian weights for yaw states."""
    print(f"Generating: w_yaw_gaussian_{int(yaw_angle_deg)}deg (Elliptical Gaussian weights)...")
    weights = np.zeros((GRID_SIZE, GRID_SIZE))
    yaw_rad = np.deg2rad(yaw_angle_deg)
    
    # Define sigma for two directions of the elliptical Gaussian
    # sigma_a: Vertical direction (major axis) standard deviation
    # sigma_b: Horizontal direction (minor axis) standard deviation
    sigma_a = radius_in_cells / 2.0
    sigma_b = (radius_in_cells * np.cos(yaw_rad)) / 2.0
    
    print(f"  Elliptical Gaussian Parameters: sigma_vert={sigma_a:.2f} cells, sigma_horiz={sigma_b:.2f} cells")
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # 2D Gaussian formula with different sigmas for axes
            x_term = ((j - center_idx)**2) / (2 * sigma_b**2)
            y_term = ((i - center_idx)**2) / (2 * sigma_a**2)
            weights[i, j] = math.exp(-(x_term + y_term))
            
    return _normalize(weights)

def generate_blade_tip_weights() -> np.ndarray:
    print("Generating: w_blade_tip (Ring weights)...")
    weights = np.zeros((GRID_SIZE, GRID_SIZE))
    r_inner = radius_in_cells * 0.8
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            dist = math.sqrt((i - center_idx)**2 + (j - center_idx)**2)
            if r_inner <= dist <= radius_in_cells:
                weights[i, j] = 1.0
    return _normalize(weights)

def generate_half_rotor_masks() -> dict:
    print("Generating: Half-region masks (w_top, w_bottom, w_left, w_right)...")
    base_mask = np.zeros((GRID_SIZE, GRID_SIZE))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if math.sqrt((i-center_idx)**2 + (j-center_idx)**2) <= radius_in_cells:
                base_mask[i,j] = 1.0
    
    masks = {}
    top = np.copy(base_mask)
    top[int(center_idx)+1:, :] = 0
    masks['w_top'] = _normalize(top)
    
    bottom = np.copy(base_mask)
    bottom[:int(center_idx), :] = 0
    masks['w_bottom'] = _normalize(bottom)
    
    left = np.copy(base_mask)
    left[:, int(center_idx)+1:] = 0
    masks['w_left'] = _normalize(left)
    
    right = np.copy(base_mask)
    right[:, :int(center_idx)] = 0
    masks['w_right'] = _normalize(right)
    
    return masks

def generate_multi_target_correlation_weights() -> np.ndarray:
    print("\n--- Starting calculation of data-driven multi-target correlation weights ---")
    all_bts_files = sorted([f for f in os.listdir(BTS_DIR) if f.endswith('.bts')])
    
    if not all_bts_files: 
        print(f"Error: No .bts files found in '{BTS_DIR}'. Cannot generate correlation weights.")
        return np.zeros((GRID_SIZE, GRID_SIZE))
        
    selected_files = all_bts_files[:NUM_CASES_FOR_CORR]
    print(f"Analyzing the first {len(selected_files)} cases.")
    
    all_wind, all_targets = [], []
    for bts_file in tqdm(selected_files, desc="Loading case data"):
        case_name = os.path.splitext(bts_file)[0]
        bts_path = os.path.join(BTS_DIR, bts_file)
        csv_path = os.path.join(TURBINE_DATA_DIR, f'{case_name}.csv')
        
        if not os.path.exists(csv_path): continue
        
        df_turbine = pd.read_csv(csv_path)
        if not all(col in df_turbine.columns for col in ['Time'] + TARGET_COLS_FOR_CORR): continue
        
        header, wind_data = read_bts_official(bts_path)
        if wind_data is None: continue
        
        # Align time indices
        indices = np.round(df_turbine['Time'].values / header['dt']).astype(int)
        if np.any(indices < 0) or np.any(indices >= header['nt']): continue
        
        all_wind.append(wind_data[indices, 0, :, :])
        all_targets.append(df_turbine[TARGET_COLS_FOR_CORR].values)
        
    if not all_wind: 
        print("Error: Failed to load any valid case data.")
        return np.zeros((GRID_SIZE, GRID_SIZE))
        
    final_wind_data = np.concatenate(all_wind, axis=0)
    final_target_data = np.concatenate(all_targets, axis=0)
    
    n_timesteps = final_wind_data.shape[0]
    wind_data_flat = final_wind_data.reshape(n_timesteps, -1)
    accumulated_corr_matrix = np.zeros((GRID_SIZE, GRID_SIZE))
    
    for i, target_name in enumerate(tqdm(TARGET_COLS_FOR_CORR, desc="Calculating target correlations")):
        target_series = final_target_data[:, i]
        target_corr_matrix = np.zeros(GRID_SIZE*GRID_SIZE)
        
        for j in range(GRID_SIZE*GRID_SIZE):
            corr, _ = stats.spearmanr(wind_data_flat[:, j], target_series)
            target_corr_matrix[j] = 0.0 if np.isnan(corr) else corr
            
        accumulated_corr_matrix += np.abs(target_corr_matrix.reshape(GRID_SIZE, GRID_SIZE))
        
    final_corr_weight = accumulated_corr_matrix / len(TARGET_COLS_FOR_CORR)
    print("Multi-target correlation weights calculation completed!")
    return _normalize(final_corr_weight)

# ==============================================================================
# 4. Main Execution Logic
# ==============================================================================
if __name__ == "__main__":
    print("--- Starting Advanced Weight Matrix Generation (V_Final - Physical Coordinates Visualization) ---")
    
    # --- Generate Weights ---
    w_hub = generate_hub_weights()
    w_yaw_gaussian = generate_yaw_gaussian_weights(YAW_ANGLE_DEG) 
    w_tip = generate_blade_tip_weights()
    half_masks = generate_half_rotor_masks()
    w_corr = generate_multi_target_correlation_weights()
    
    weights_dict = {
        'w_hub': w_hub,
        f'w_yaw_gaussian_{int(YAW_ANGLE_DEG)}deg': w_yaw_gaussian,
        'w_blade_tip': w_tip,
        'w_corr_multi_target': w_corr,
        **half_masks
    }
    
    # --- Save Files ---
    print("\n--- Saving weight matrices as .npy files ---")
    for name, matrix in weights_dict.items():
        filepath = os.path.join(OUTPUT_DIR, f"{name}.npy")
        np.save(filepath, matrix)
        print(f"Saved: {filepath}")
        
    # --- Visualization ---
    print("\n--- Generating combined visualization with physical coordinates ---")
    num_weights = len(weights_dict)
    cols = 4
    rows = math.ceil(num_weights / cols)
    
    # Adjust plot size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5)) 
    axes = axes.flatten()

    plot_order = [
        'w_hub', f'w_yaw_gaussian_{int(YAW_ANGLE_DEG)}deg', 'w_blade_tip', 'w_corr_multi_target',
        'w_top', 'w_bottom', 'w_left', 'w_right'
    ]
    
    # 1. Create physical coordinates
    physical_coords = np.linspace(0, PLANE_SIZE_M, GRID_SIZE)
    
    for i, name in enumerate(plot_order):
        if name in weights_dict:
            ax = axes[i]
            matrix = weights_dict[name]
            
            # 2. Wrap numpy matrix in a DataFrame with physical coordinates
            df_to_plot = pd.DataFrame(
                matrix,
                index=physical_coords,
                columns=physical_coords
            )
            
            # 3. Plot using DataFrame and control label density
            sns.heatmap(
                df_to_plot, 
                ax=ax, 
                cmap='viridis', 
                square=True, 
                cbar=True,
                xticklabels=5,  # Show one label every 5 ticks
                yticklabels=5
            )
            
            # 4. Update axes titles and labels
            ax.set_title(name, fontsize=16)
            ax.set_xlabel('Horizontal Position (m)', fontsize=12)
            ax.set_ylabel('Vertical Position (m)', fontsize=12)
            
            # Rotate x-axis labels to prevent overlap
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Keep Y-axis origin at the bottom-left
            ax.invert_yaxis()

    # Hide unused subplots
    for j in range(len(plot_order), len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout(pad=1.5) 
    output_path_png = os.path.join(OUTPUT_DIR, 'combined_weights_visualization_final.png')
    plt.savefig(output_path_png, dpi=200)
    print(f"\nCombined visualization saved to: {output_path_png}")
    plt.show()