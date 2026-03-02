# --- 增强版特征提取器 PIFENet ---

import os
import torch
import torch.nn as nn
import numpy as np



# --- 测试函数 ---git add main_solver.py
def test_v8_extractor():
    """测试 V8 版本的特征提取器"""
    WEIGHTS_DIR = 'precomputed_weights_v3'
    if not os.path.exists(WEIGHTS_DIR):
        print(f"错误: 找不到权重目录 '{WEIGHTS_DIR}'。请先运行权重生成脚本。")
        # 为方便测试，如果目录不存在则创建虚拟权重文件
        print("正在创建虚拟权重文件用于测试...")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        weight_files = {
            'w_hub.npy': np.ones((31, 31)), 'w_yaw_gaussian_60deg.npy': np.ones((31, 31)),
            'w_blade_tip.npy': np.ones((31, 31)), 'w_corr_multi_target.npy': np.ones((31, 31)),
            'w_top.npy': np.ones((31, 31)), 'w_bottom.npy': np.ones((31, 31)),
            'w_left.npy': np.ones((31, 31)), 'w_right.npy': np.ones((31, 31))
        }
        for filename, arr in weight_files.items():
            np.save(os.path.join(WEIGHTS_DIR, filename), arr)
        print("虚拟权重文件创建完毕。")


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
        
        # 按组显示特征
        groups = [
            ("全局基础特征", 0, 10),
            ("权重区域特征", 10, 28),
            ("湍流特性特征", 28, 34),
            ("空间结构特征", 34, 39),
            ("极值统计特征", 39, 42),
            ("能量分布特征", 42, 45)
        ]
        
        for group_name, start, end in groups:
            print(f"\n{group_name} ({end-start}个):")
            for i in range(start, end):
                if i < len(model.feature_names):
                    is_finite = torch.all(torch.isfinite(features[:, i])).item()
                    status = "✓ OK" if is_finite else "✗ NaN/Inf"
                    print(f"  {i+1:02d}: {model.feature_names[i]:<35} {status}")
        
        print("\n" + "=" * 80)
        print("特征统计:")
        num_valid_features = torch.sum(torch.all(torch.isfinite(features), dim=0)).item()
        print(f"- 总特征数: {len(model.feature_names)}")
        print(f"- 有效特征数 (无NaN/Inf): {num_valid_features}")
        if num_valid_features > 0:
            valid_features = features[:, torch.all(torch.isfinite(features), dim=0)]
            print(f"- 特征范围: [{valid_features.min().item():.4f}, {valid_features.max().item():.4f}]")
            print(f"- 特征均值: {valid_features.mean().item():.4f}")
            print(f"- 特征标准差: {valid_features.std().item():.4f}")
        
        if num_valid_features < len(model.feature_names):
            print("\n警告：部分特征计算结果为 NaN 或 Inf，请检查相关实现。")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_v8_extractor()
