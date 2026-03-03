"""TurbSim .bts reader (official-compatible)."""
import struct
import numpy as np
import pprint

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

if __name__ == "__main__":
    file_to_read = 'wind.bts'
    header_info, wind_data = read_bts_official(file_to_read)

    if header_info and wind_data is not None:
        print("--- File Header Info (Parsed based on official code) ---")
        # Print only some useful header info
        print(f"  ID: {header_info['ID']}")
        print(f"  Dimensions (t, y, z): ({header_info['nt']}, {header_info['ny']}, {header_info['nz']})")
        print(f"  uHub: {header_info['uHub']:.2f} m/s, zHub: {header_info['zHub']:.2f} m")
        print(f"  Info: {header_info['info']}")
        
        print("\n" + "="*50 + "\n")
        print("--- Wind Speed Data Summary (Parsed based on official code) ---")
        print(f"Data Array Shape: {wind_data.shape} (Time Steps, Components, Y points, Z points)")
        
        u_mean = np.mean(wind_data[:, 0, :, :])
        v_mean = np.mean(wind_data[:, 1, :, :])
        w_mean = np.mean(wind_data[:, 2, :, :])
        
        print(f"\nU-Component Mean: {u_mean:.4f} m/s (Should be close to uHub={header_info['uHub']:.2f})")
        print(f"V-Component Mean: {v_mean:.4f} m/s (Should be close to 0)")
        print(f"W-Component Mean: {w_mean:.4f} m/s (Should be close to 0)")
        
        # Example data point
        t, y, z = 0, header_info['ny'] // 2, header_info['nz'] // 2
        u_pt, v_pt, w_pt = wind_data[t, 0, y, z], wind_data[t, 1, y, z], wind_data[t, 2, y, z]
        
        print(f"\nExample Data Point (t=0, Grid Center): U={u_pt:.4f}, V={v_pt:.4f}, W={w_pt:.4f} m/s")