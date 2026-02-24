"""TurbSim .bts reader (official-compatible)."""
import struct
import numpy as np
import pprint

def read_bts_official(filepath):
    """
    读取并解析 TurbSim .bts 文件。
    此版本严格遵循用户提供的官方 'turbsim_file.py' 的逻辑。
    """
    try:
        with open(filepath, 'rb') as f:
            # --- 1. 读取文件头 (根据官方代码) ---
            header = {}
            
            # 读取 ID, 维度, 和时间步 (2 + 4*4 = 18字节)
            (header['ID'], 
             header['nz'], 
             header['ny'], 
             header['nTwr'], 
             header['nt']) = struct.unpack('<h4l', f.read(18))

            # 读取间距和参考值 (6*4 = 24字节)
            (header['dz'], 
             header['dy'], 
             header['dt'], 
             header['uHub'], 
             header['zHub'], 
             header['zBottom']) = struct.unpack('<6f', f.read(24))

            # 读取 U, V, W 的缩放和偏移因子 (6*4 = 24字节)
            scl_off = struct.unpack('<6f', f.read(24))
            scl = np.float32([scl_off[0], scl_off[2], scl_off[4]])
            off = np.float32([scl_off[1], scl_off[3], scl_off[5]])
            header['scale_factors'] = scl
            header['offsets'] = off
            
            # 读取描述字符串的长度 (4字节整数)
            (nChar,) = struct.unpack('<l', f.read(4))
            
            # 读取描述字符串本身
            header['info'] = f.read(nChar).decode('utf-8', errors='ignore').strip()

            # --- 2. 读取风速数据 ---
            # 根据官方代码，它在一个循环中按时间步读取
            # 这样可以正确处理交错的网格和塔筒数据
            
            ny, nz, nt, nTwr = header['ny'], header['nz'], header['nt'], header['nTwr']
            
            # 初始化空的 numpy 数组来存放数据
            u = np.zeros((3, nt, ny, nz), dtype=np.float32)
            uTwr = np.zeros((3, nt, nTwr), dtype=np.float32)

            for it in range(nt):
                # 读取主网格数据
                # 2字节(int16) * 3个分量 * ny * nz
                grid_bytes = 2 * 3 * ny * nz
                if grid_bytes > 0:
                    buffer_grid = np.frombuffer(f.read(grid_bytes), dtype=np.int16)
                    # 使用 Fortran-style (order='F') 重塑，这很关键
                    reshaped_grid = buffer_grid.astype(np.float32).reshape((3, ny, nz), order='F')
                    u[:, it, :, :] = reshaped_grid
                
                # 读取塔筒数据
                # 2字节(int16) * 3个分量 * nTwr
                tower_bytes = 2 * 3 * nTwr
                if tower_bytes > 0:
                    buffer_tower = np.frombuffer(f.read(tower_bytes), dtype=np.int16)
                    reshaped_tower = buffer_tower.astype(np.float32).reshape((3, nTwr), order='F')
                    uTwr[:, it, :] = reshaped_tower

            # --- 3. 应用缩放和偏移 (根据官方代码) ---
            # 公式: (原始值 - offset) / scale
            # 使用广播 (broadcasting) 来高效计算
            u -= off[:, None, None, None]
            u /= scl[:, None, None, None]

            if nTwr > 0:
                uTwr -= off[:, None, None]
                uTwr /= scl[:, None, None]
            
            # 为了方便使用，我们将数据轴顺序调整为 (nt, 3, ny, nz)
            wind_data = np.transpose(u, (1, 0, 2, 3))
            
            return header, wind_data

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None, None

if __name__ == "__main__":
    file_to_read = 'wind.bts'
    header_info, wind_data = read_bts_official(file_to_read)

    if header_info and wind_data is not None:
        print("--- 文件头信息 (基于官方代码解析) ---")
        # 只打印部分有用的头信息
        print(f"  ID: {header_info['ID']}")
        print(f"  Dimensions (t, y, z): ({header_info['nt']}, {header_info['ny']}, {header_info['nz']})")
        print(f"  uHub: {header_info['uHub']:.2f} m/s, zHub: {header_info['zHub']:.2f} m")
        print(f"  Info: {header_info['info']}")
        
        print("\n" + "="*50 + "\n")
        print("--- 风速数据摘要 (基于官方代码解析) ---")
        print(f"数据数组形状: {wind_data.shape} (时间步, 分量, Y点数, Z点数)")
        
        u_mean = np.mean(wind_data[:, 0, :, :])
        v_mean = np.mean(wind_data[:, 1, :, :])
        w_mean = np.mean(wind_data[:, 2, :, :])
        
        print(f"\nU分量平均值: {u_mean:.4f} m/s (应接近 uHub={header_info['uHub']:.2f})")
        print(f"V分量平均值: {v_mean:.4f} m/s (应接近 0)")
        print(f"W分量平均值: {w_mean:.4f} m/s (应接近 0)")
        
        # 示例数据点
        t, y, z = 0, header_info['ny'] // 2, header_info['nz'] // 2
        u_pt, v_pt, w_pt = wind_data[t, 0, y, z], wind_data[t, 1, y, z], wind_data[t, 2, y, z]
        
        print(f"\n示例数据点 (t=0, 网格中心): U={u_pt:.4f}, V={v_pt:.4f}, W={w_pt:.4f} m/s")
