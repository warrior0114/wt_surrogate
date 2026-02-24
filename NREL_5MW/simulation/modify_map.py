import struct
import numpy as np
import pprint
import time

# ==============================================================================
# 1. .bts 文件读取函数 (官方标准版)
# ==============================================================================
def read_bts_official(filepath):
    """
    读取并解析 TurbSim .bts 文件。
    此版本严格遵循官方 'turbsim_file.py' 定义的格式。
    """
    try:
        with open(filepath, 'rb') as f:
            header = {}
            # 读取 ID, 维度, 和时间步 (2 + 4*4 = 18字节)
            (header['ID'], 
             header['nz'], header['ny'], 
             header['nTwr'], header['nt']) = struct.unpack('<h4l', f.read(18))

            if header['ID'] != 7: raise IOError("文件格式不正确：ID不为7")

            # 读取间距和参考值 (6*4 = 24字节)
            (header['dz'], header['dy'], header['dt'], 
             header['uHub'], header['zHub'], header['zBottom']) = struct.unpack('<6f', f.read(24))

            # 读取 U, V, W 的缩放和偏移因子 (6*4 = 24字节)
            scl_off = struct.unpack('<6f', f.read(24))
            scl = np.float32([scl_off[0], scl_off[2], scl_off[4]])
            off = np.float32([scl_off[1], scl_off[3], scl_off[5]])
            
            # 读取描述字符串的长度 (4字节整数)
            (nChar,) = struct.unpack('<l', f.read(4))
            # 读取描述字符串本身
            header['info'] = f.read(nChar).decode('utf-8', errors='ignore').strip()

            # 按时间步循环读取数据，以正确处理交错的网格和塔筒数据
            ny, nz, nt, nTwr = header['ny'], header['nz'], header['nt'], header['nTwr']
            
            u_raw_all = np.zeros((3, nt, ny, nz), dtype=np.float32)

            for it in range(nt):
                # 主网格数据
                grid_bytes = 2 * 3 * ny * nz
                if grid_bytes > 0:
                    buffer_grid = np.frombuffer(f.read(grid_bytes), dtype=np.int16)
                    # 使用 Fortran-style (order='F') 重塑，这很关键
                    reshaped_grid = buffer_grid.astype(np.float32).reshape((3, ny, nz), order='F')
                    u_raw_all[:, it, :, :] = reshaped_grid
                
                # 跳过塔筒数据
                tower_bytes = 2 * 3 * nTwr
                if tower_bytes > 0:
                    f.read(tower_bytes)
            
            # 应用缩放和偏移
            u_physical = (u_raw_all - off[:, None, None, None]) / scl[:, None, None, None]
            
            # 调整轴顺序为 (nt, 3, ny, nz) 以便后续处理
            wind_data = np.transpose(u_physical, (1, 0, 2, 3))
            
            return header, wind_data

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None, None

# ==============================================================================
# 2. 向量化的地形修正函数
# ==============================================================================

def precompute_terrain_modifiers(slope_angle_deg, grid_size=31):
    """
    预计算地形修正所需的共享参数（加速矩阵和旋转角度）。
    """
    if np.isclose(slope_angle_deg, 0.0):
        return None  # 无坡度，无需修正

    slope_rad = np.radians(slope_angle_deg)
    hub_height_index = grid_size // 2
    z_coords = np.arange(grid_size) - hub_height_index
    max_speed_up_factor = 1.6 * slope_rad
    terrain_length_scale = float(grid_size)
    speed_up_profile = 1.0 + max_speed_up_factor * np.exp(-2.5 * (z_coords - z_coords.min()) / terrain_length_scale)
    speed_up_matrix = np.tile(speed_up_profile, (grid_size, 1)).T
    
    return {
        "speed_up_matrix": speed_up_matrix,
        "cos_slope": np.cos(slope_rad),
        "sin_slope": np.sin(slope_rad)
    }

def apply_terrain_correction_vectorized(wind_data, modifiers):
    """
    【向量化版本】对整个四维风场数据应用地形修正，无需 for 循环。

    Args:
        wind_data (np.array): 原始风场数据，形状为 (nt, 3, ny, nz)。
        modifiers (dict): 从 precompute_terrain_modifiers() 获得的参数字典。

    Returns:
        np.array: 修正后的四维风场数据。
    """
    if modifiers is None:
        return wind_data

    # 提取参数
    speed_up_matrix = modifiers["speed_up_matrix"] # shape: (ny, nz)
    cos_slope = modifiers["cos_slope"]
    sin_slope = modifiers["sin_slope"]

    # 提取原始风速分量
    U_orig = wind_data[:, 0, :, :] # shape: (nt, ny, nz)
    V_orig = wind_data[:, 1, :, :] # shape: (nt, ny, nz)
    W_orig = wind_data[:, 2, :, :] # shape: (nt, ny, nz)

    # 1. 应用加速效应 (利用 NumPy广播机制)
    # (nt, ny, nz) * (ny, nz) -> (nt, ny, nz)
    U_accelerated = U_orig * speed_up_matrix
    V_final = V_orig * speed_up_matrix
    
    # 2. 应用旋转效应 (矩阵与标量运算，天然向量化)
    U_final = U_accelerated * cos_slope - W_orig * sin_slope
    W_final = U_accelerated * sin_slope + W_orig * cos_slope
    
    # 3. 组装回四维数组
    modified_data = np.copy(wind_data) # 使用 copy 避免修改原始数据
    modified_data[:, 0, :, :] = U_final
    modified_data[:, 1, :, :] = V_final
    modified_data[:, 2, :, :] = W_final
    
    return modified_data
# 写入文件
def write_bts_official(filepath, header, wind_data):
    """
    将修正后的风场数据写入一个新的 .bts 文件。

    Args:
        filepath (str): 输出文件的路径。
        header (dict): 从原始文件读取的头信息字典。
        wind_data (np.array): 修正后的四维风场数据 (nt, 3, ny, nz)。
    """
    print(f"正在准备写入文件: {filepath}...")
    
    # --- 步骤 A: 准备数据和头信息 ---
    # 将数据从 (nt, 3, ny, nz) 转回 (3, nt, ny, nz) 以便处理
    u_physical = np.transpose(wind_data, (1, 0, 2, 3))
    
    # 定义 int16 的范围
    int_min, int_max = -32768, 32767
    int_range = int_max - int_min # 65535

    new_scl = np.zeros(3, dtype=np.float32)
    new_off = np.zeros(3, dtype=np.float32)
    u_int16 = np.zeros_like(u_physical, dtype=np.int16)

    # --- 步骤 B: 为每个分量重新计算 scale 和 offset ---
    for i in range(3): # 循环处理 U, V, W
        min_val = u_physical[i].min()
        max_val = u_physical[i].max()
        
        if np.isclose(min_val, max_val):
            new_scl[i] = 1.0
        else:
            new_scl[i] = int_range / (max_val - min_val)
            
        # 根据官方代码 `off[k] = intmin - scl[k] * all_min` 计算新的 offset
        new_off[i] = int_min - new_scl[i] * min_val
        
        # 将浮点数数据转换为 int16
        # 公式: int_val = float_val * scale + offset
        scaled_data = u_physical[i] * new_scl[i] + new_off[i]
        u_int16[i] = np.round(scaled_data).astype(np.int16)

    # --- 步骤 C: 准备新的描述信息 ---
    new_info_str = f"Original file '{header.get('info', 'unknown')}' modified with terrain slope on {time.strftime('%Y-%m-%d %H:%M:%S')}."
    new_info_bytes = new_info_str.encode('utf-8')
    nChar = len(new_info_bytes)

    # --- 步骤 D: 写入文件 ---
    try:
        with open(filepath, 'wb') as f:
            # 1. 写入 ID 和维度
            f.write(struct.pack('<h4l', header['ID'], header['nz'], header['ny'], header['nTwr'], header['nt']))
            
            # 2. 写入间距和参考值 (使用原始值)
            f.write(struct.pack('<6f', header['dz'], header['dy'], header['dt'], header['uHub'], header['zHub'], header['zBottom']))
            
            # 3. 写入新的 scale 和 offset
            scl_off_pairs = [val for pair in zip(new_scl, new_off) for val in pair]
            f.write(struct.pack('<6f', *scl_off_pairs))
            
            # 4. 写入新的描述信息
            f.write(struct.pack('<l', nChar))
            f.write(new_info_bytes)
            
            # 5. 按时间步写入数据
            nt, nTwr = header['nt'], header['nTwr']
            for it in range(nt):
                # 写入主网格数据
                # 使用 tobytes(order='F') 来确保 Fortran-style 的内存布局
                f.write(u_int16[:, it, :, :].tobytes(order='F'))
                
                # 写入空的塔筒数据 (如果原始文件中有)
                if nTwr > 0:
                    empty_tower_data = np.zeros((3, nTwr), dtype=np.int16)
                    f.write(empty_tower_data.tobytes(order='F'))
        
        print(f"成功将修正后的数据保存到: {filepath}")

    except Exception as e:
        print(f"写入文件时发生错误: {e}")
    return filepath


def modify_wind_data(input_bts_file,output_bts_file,slope_angle):
    header_info, original_wind_data = read_bts_official(input_bts_file)
    if original_wind_data is not None:
        grid_size = header_info['ny']
        print(f'print(f"\n应用坡度为 {slope_angle}° 的地形修正 (向量化版本)...")')
        terrain_modifiers = precompute_terrain_modifiers(slope_angle, grid_size)
        modified_wind_data = apply_terrain_correction_vectorized(original_wind_data, terrain_modifiers)
        output_bts_file = write_bts_official(output_bts_file, header_info, modified_wind_data)
        print('------------successful modify map -------------------')
    return output_bts_file

# ==============================================================================
# 3. 主程序
# ==============================================================================

if __name__ == "__main__":
    # --- 用户设置 ---
    file_to_read = 'wind.bts'
    slope_angle = 5.0  # <--- 在这里设置地形坡度 (单位: 度)
    output_filename = 'wind_modified_slope_5.0deg.bts'

    # --- 步骤 1: 读取原始风场数据 ---
    print(f"正在读取文件: {file_to_read}...")
    header_info, original_wind_data = read_bts_official(file_to_read)

    if original_wind_data is not None:
        grid_size = header_info['ny']
        
        # --- 步骤 2: 应用地形修正 ---
        print(f"\n应用坡度为 {slope_angle}° 的地形修正 (向量化版本)...")
        
        # 预计算修正参数
        terrain_modifiers = precompute_terrain_modifiers(slope_angle, grid_size)
        print('修正矩阵',terrain_modifiers)
        
        # 调用向量化函数，一步完成所有时间步的修正
        start_time = time.time()
        modified_wind_data = apply_terrain_correction_vectorized(original_wind_data, terrain_modifiers)
        end_time = time.time()
        
        print(f"地形修正完成，耗时: {end_time - start_time:.4f} 秒")
        write_bts_official(output_filename, header_info, modified_wind_data)

        # --- 步骤 3: 结果对比分析 ---
        print("\n" + "="*50 + "\n")
        print("--- 结果对比分析 ---")

        # 提取网格中心点进行对比
        y_mid, z_mid = grid_size // 2, grid_size // 2
        
        orig_u_mean = np.mean(original_wind_data[:, 0, y_mid, z_mid])
        orig_w_mean = np.mean(original_wind_data[:, 2, y_mid, z_mid])
        
        mod_u_mean = np.mean(modified_wind_data[:, 0, y_mid, z_mid])
        mod_w_mean = np.mean(modified_wind_data[:, 2, y_mid, z_mid])
        
        print(f"网格中心点 (y={y_mid}, z={z_mid}) 的平均风速对比:")
        print(f"  原始 U 平均值: {orig_u_mean:.4f} m/s")
        print(f"  修正后 U 平均值: {mod_u_mean:.4f} m/s  <-- 应该略有增加 (加速效应)")
        print("-" * 20)
        print(f"  原始 W 平均值: {orig_w_mean:.4f} m/s  <-- 应该接近 0")
        print(f"  修正后 W 平均值: {mod_w_mean:.4f} m/s  <-- 应该变为正值 (抬升效应)")

        # 理论验证
        expected_w_mean_from_rotation = orig_u_mean * np.sin(np.radians(slope_angle))
        print(f"  理论 W 平均值 (仅考虑旋转效应): {expected_w_mean_from_rotation:.4f} m/s")
        print("\n修正后的数据已保存在变量 'modified_wind_data' 中，可用于后续处理。")