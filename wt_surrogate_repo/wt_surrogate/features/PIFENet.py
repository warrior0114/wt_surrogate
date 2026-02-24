# --- 增强版特征提取器 PIFENet ---

import os
import torch
import torch.nn as nn
import numpy as np

class PIFENet(nn.Module):
    """
    增强版特征提取网络 (V8) - 扩展到45个特征
    
    在保持原有权重矩阵不变的基础上，通过增加新的特征计算方法
    来提取更多样化的风场状态特征，涵盖：
    - 湍流特性：多尺度湍流强度、频谱特征
    - 空间结构：梯度、曲率、各向异性
    - 极值统计：偏度、峰度、极值比
    - 能量分布：动能、势能、剪切能量
    - 相干性：空间相关性、周期性特征
    """
    
    def __init__(self, weights_dir: str):
        super(PIFENet, self).__init__()
        self.weights_dir = weights_dir
        
        self.weight_files = {
            'w_hub': 'w_hub.npy', 'w_yaw': 'w_yaw_gaussian_60deg.npy',
            'w_tip': 'w_blade_tip.npy', 'w_corr': 'w_corr_multi_target.npy',
            'w_top': 'w_top.npy', 'w_bottom': 'w_bottom.npy',
            'w_left': 'w_left.npy', 'w_right': 'w_right.npy'
        }
        
        self._load_and_register_weights()
        self.feature_names = self._get_feature_names()
        
        print(f"PIFENet initialized with {len(self.feature_names)} features.")
        assert len(self.feature_names) == 45, f"Expected 45 features, but got {len(self.feature_names)}."

    def _load_and_register_weights(self):
        print("Loading pre-computed weights for V8 extractor...")
        for name, filename in self.weight_files.items():
            path = os.path.join(self.weights_dir, filename)
            if not os.path.exists(path): 
                raise FileNotFoundError(f"Weight file not found: {path}.")
            tensor = torch.from_numpy(np.load(path)).float()
            self.register_buffer(name.replace('w_',''), tensor)
        print("All weights loaded and registered successfully.")

    def _get_feature_names(self) -> list[str]:
        names = []
        
        # Group 1: 全局基础特征 (10个) - 保持不变
        names.extend(['global_u_mean', 'global_v_mean', 'global_w_mean'])
        names.extend(['global_u_std', 'global_v_std', 'global_w_std'])
        names.extend(['global_tke', 'global_speed_max', 'global_gust_factor', 'global_grad_norm'])
        
        # Group 2: 权重区域特征 (18个) - 保持不变
        # 2.1 Hub (2)
        names.extend(['hub_u_mean', 'hub_tke'])
        # 2.2 Yaw (2)
        names.extend(['yaw_u_mean', 'yaw_u_std'])
        # 2.3 Tip (2)
        names.extend(['tip_u_mean', 'tip_speed_range'])
        # 2.4 Corr (5)
        names.extend(['corr_u_mean', 'corr_tke', 'corr_ti_u', 'corr_flow_inclination', 'corr_flow_direction'])
        # 2.5 Vertical (3)
        names.extend(['vertical_shear_u', 'vertical_shear_w', 'vertical_shear_norm'])
        # 2.6 Lateral (4)
        names.extend(['lateral_asymmetry_u', 'lateral_asymmetry_v', 'lateral_tke_diff', 'lateral_asymmetry_speed'])
        
        # Group 3: 新增湍流特性特征 (6个)
        names.extend(['global_ti_total', 'global_anisotropy_ratio', 'global_reynolds_stress'])
        names.extend(['hub_ti_total', 'tip_vorticity_strength', 'corr_turbulence_ratio'])
        
        # Group 4: 新增空间结构特征 (5个)
        names.extend(['global_curvature_mean', 'global_laplacian_norm', 'spatial_heterogeneity'])
        names.extend(['vertical_gradient_ratio', 'lateral_gradient_ratio'])
        
        # Group 5: 新增极值统计特征 (3个)
        names.extend(['global_skewness_u', 'global_kurtosis_u', 'extreme_wind_ratio'])
        
        # Group 6: 新增能量分布特征 (3个)
        names.extend(['kinetic_energy_density', 'shear_production_rate', 'energy_cascade_indicator'])
        
        return names

    def _weighted_mean(self, tensor, weights):
        return torch.sum(tensor * weights.unsqueeze(0), dim=(1, 2))

    def _weighted_std(self, tensor, weights):
        mean = self._weighted_mean(tensor, weights).view(-1, 1, 1)
        var = torch.sum(weights.unsqueeze(0) * (tensor - mean)**2, dim=(1, 2))
        return torch.sqrt(var)
        
    def _weighted_tke(self, u, v, w, weights):
        u_std = self._weighted_std(u, weights)
        v_std = self._weighted_std(v, weights)
        w_std = self._weighted_std(w, weights)
        return 0.5 * (u_std**2 + v_std**2 + w_std**2)
    
    def _weighted_skewness(self, tensor, weights):
        """计算加权偏度"""
        mean = self._weighted_mean(tensor, weights).view(-1, 1, 1)
        std = self._weighted_std(tensor, weights).view(-1, 1, 1)
        skew = torch.sum(weights.unsqueeze(0) * ((tensor - mean) / (std + 1e-8))**3, dim=(1, 2))
        return skew
    
    def _weighted_kurtosis(self, tensor, weights):
        """计算加权峰度"""
        mean = self._weighted_mean(tensor, weights).view(-1, 1, 1)
        std = self._weighted_std(tensor, weights).view(-1, 1, 1)
        kurt = torch.sum(weights.unsqueeze(0) * ((tensor - mean) / (std + 1e-8))**4, dim=(1, 2))
        return kurt - 3.0  # 超峰度

    def forward(self, wind_matrix: torch.Tensor) -> torch.Tensor:
        u, v, w = wind_matrix[:, 0], wind_matrix[:, 1], wind_matrix[:, 2]
        speed = torch.sqrt(u**2 + v**2 + w**2 + 1e-8)
        features = {}
        
        # ===== Group 1: 全局基础特征 (10个) - 保持不变 =====
        features['global_u_mean'] = torch.mean(u, dim=(1, 2))
        features['global_v_mean'] = torch.mean(v, dim=(1, 2))
        features['global_w_mean'] = torch.mean(w, dim=(1, 2))
        
        global_u_std = torch.std(u, dim=(1, 2))
        global_v_std = torch.std(v, dim=(1, 2))
        global_w_std = torch.std(w, dim=(1, 2))
        features['global_u_std'] = global_u_std
        features['global_v_std'] = global_v_std
        features['global_w_std'] = global_w_std
        
        features['global_tke'] = 0.5 * (global_u_std**2 + global_v_std**2 + global_w_std**2)
        features['global_speed_max'] = torch.amax(speed, dim=(1, 2))
        
        global_speed_mean = torch.mean(speed, dim=(1, 2))
        features['global_gust_factor'] = features['global_speed_max'] / (global_speed_mean + 1e-8)
        
        # 梯度范数
        grad_y = torch.mean(torch.abs(torch.diff(speed, dim=1)), dim=(1, 2))
        grad_z = torch.mean(torch.abs(torch.diff(speed, dim=2)), dim=(1, 2))
        features['global_grad_norm'] = torch.sqrt(grad_y**2 + grad_z**2)
        
        # ===== Group 2: 权重区域特征 (18个) - 保持不变 =====
        # Hub特征
        features['hub_u_mean'] = self._weighted_mean(u, self.hub)
        features['hub_tke'] = self._weighted_tke(u, v, w, self.hub)
        
        # Yaw特征
        features['yaw_u_mean'] = self._weighted_mean(u, self.yaw)
        features['yaw_u_std'] = self._weighted_std(u, self.yaw)
        
        # Tip特征
        features['tip_u_mean'] = self._weighted_mean(u, self.tip)
        tip_mask = (self.tip > 0).float()
        masked_speed = speed * tip_mask.unsqueeze(0)
        tip_speed_max = torch.amax(masked_speed, dim=(1, 2))
        tip_speed_min = torch.amin(torch.where(masked_speed > 0, masked_speed, 1e9), dim=(1, 2))
        features['tip_speed_range'] = tip_speed_max - tip_speed_min
        
        # Corr特征
        corr_u_mean = self._weighted_mean(u, self.corr)
        features['corr_u_mean'] = corr_u_mean
        features['corr_tke'] = self._weighted_tke(u, v, w, self.corr)
        corr_u_std = self._weighted_std(u, self.corr)
        features['corr_ti_u'] = corr_u_std / (corr_u_mean + 1e-8)
        
        corr_v_mean = self._weighted_mean(v, self.corr)
        corr_w_mean = self._weighted_mean(w, self.corr)
        features['corr_flow_inclination'] = torch.atan2(corr_w_mean, corr_u_mean)
        features['corr_flow_direction'] = torch.atan2(corr_v_mean, corr_u_mean)
        
        # Vertical特征
        vertical_shear_u = self._weighted_mean(u, self.top) - self._weighted_mean(u, self.bottom)
        features['vertical_shear_u'] = vertical_shear_u
        features['vertical_shear_w'] = self._weighted_mean(w, self.top) - self._weighted_mean(w, self.bottom)
        features['vertical_shear_norm'] = vertical_shear_u / (corr_u_mean + 1e-8)
        
        # Lateral特征
        features['lateral_asymmetry_u'] = self._weighted_mean(u, self.left) - self._weighted_mean(u, self.right)
        features['lateral_asymmetry_v'] = self._weighted_mean(v, self.left) - self._weighted_mean(v, self.right)
        tke_left = self._weighted_tke(u, v, w, self.left)
        tke_right = self._weighted_tke(u, v, w, self.right)
        features['lateral_tke_diff'] = tke_left - tke_right
        features['lateral_asymmetry_speed'] = self._weighted_mean(speed, self.left) - self._weighted_mean(speed, self.right)
        
        # ===== Group 3: 新增湍流特性特征 (6个) =====
        # 总湍流强度
        total_std = torch.sqrt(global_u_std**2 + global_v_std**2 + global_w_std**2)
        features['global_ti_total'] = total_std / (global_speed_mean + 1e-8)
        
        # 各向异性比 (最大std与最小std的比值)
        std_max = torch.maximum(torch.maximum(global_u_std, global_v_std), global_w_std)
        std_min = torch.minimum(torch.minimum(global_u_std, global_v_std), global_w_std)
        features['global_anisotropy_ratio'] = std_max / (std_min + 1e-8)
        
        # Reynolds应力 (u'v'的估计)
        u_fluct = u - features['global_u_mean'].view(-1, 1, 1)
        v_fluct = v - features['global_v_mean'].view(-1, 1, 1)
        features['global_reynolds_stress'] = torch.mean(u_fluct * v_fluct, dim=(1, 2))
        
        # Hub区域总湍流强度
        hub_u_std = self._weighted_std(u, self.hub)
        hub_v_std = self._weighted_std(v, self.hub)
        hub_w_std = self._weighted_std(w, self.hub)
        hub_total_std = torch.sqrt(hub_u_std**2 + hub_v_std**2 + hub_w_std**2)
        features['hub_ti_total'] = hub_total_std / (features['hub_u_mean'] + 1e-8)
        
        # 叶尖区域涡量强度 (使用速度梯度估计)
        tip_speed = speed * tip_mask.unsqueeze(0)
        tip_grad_y = torch.mean(torch.abs(torch.diff(tip_speed, dim=1)), dim=(1, 2))
        tip_grad_z = torch.mean(torch.abs(torch.diff(tip_speed, dim=2)), dim=(1, 2))
        features['tip_vorticity_strength'] = torch.sqrt(tip_grad_y**2 + tip_grad_z**2)
        
        # 相关区域湍流比 (TKE与平均动能的比值)
        corr_mean_ke = 0.5 * (corr_u_mean**2 + self._weighted_mean(v, self.corr)**2 + corr_w_mean**2)
        features['corr_turbulence_ratio'] = features['corr_tke'] / (corr_mean_ke + 1e-8)
        
        # ===== Group 4: 新增空间结构特征 (5个) =====
        # 曲率 (二阶导数的估计)
        u_laplacian = torch.diff(torch.diff(u, dim=1), dim=1)
        v_laplacian = torch.diff(torch.diff(v, dim=1), dim=1)
        curvature = torch.sqrt(u_laplacian**2 + v_laplacian**2 + 1e-8)
        features['global_curvature_mean'] = torch.mean(curvature, dim=(1, 2))
        
        # Laplacian范数 (散度相关)
        # --- FIX START: 修复维度不匹配的BUG ---
        # 分别计算两个方向的二阶导数
        u_yy = torch.diff(u, n=2, dim=1)  # Shape: (B, H-2, W) -> (B, 29, 31)
        u_zz = torch.diff(u, n=2, dim=2)  # Shape: (B, H, W-2) -> (B, 31, 29)
        # 裁剪到共同的内部区域(H-2, W-2)再相加
        laplacian_u = u_yy[:, :, 1:-1] + u_zz[:, 1:-1, :] # Both are now (B, 29, 29)
        # --- FIX END ---
        features['global_laplacian_norm'] = torch.mean(torch.abs(laplacian_u), dim=(1, 2))
        
        # 空间异质性 (局部方差的方差)
        kernel_size = 3
        # 使用Unfold + Var实现更高效的局部方差计算
        patches = torch.nn.functional.unfold(speed.unsqueeze(1), kernel_size=(kernel_size, kernel_size), padding=kernel_size//2).transpose(1, 2)
        local_vars_flat = torch.var(patches, dim=2)
        H_out, W_out = speed.shape[1], speed.shape[2]
        local_vars = local_vars_flat.reshape(speed.shape[0], H_out, W_out)
        features['spatial_heterogeneity'] = torch.std(local_vars, dim=(1, 2))
        
        # 垂直梯度比
        # --- FIX START: 修复计算逻辑，避免NaN ---
        # 先计算整个场的垂直梯度(沿z轴, dim=2)，然后取加权平均
        u_grad_z = torch.abs(torch.diff(u, dim=2, prepend=u[:, :, :1]))
        top_grad = self._weighted_mean(u_grad_z, self.top)
        bottom_grad = self._weighted_mean(u_grad_z, self.bottom)
        features['vertical_gradient_ratio'] = (top_grad + 1e-8) / (bottom_grad + 1e-8)
        # --- FIX END ---
        
        # 水平梯度比
        left_u_mean = self._weighted_mean(u, self.left)
        right_u_mean = self._weighted_mean(u, self.right)
        features['lateral_gradient_ratio'] = torch.abs(left_u_mean - right_u_mean) / (features['global_u_mean'] + 1e-8)
        
        # ===== Group 5: 新增极值统计特征 (3个) =====
        # 使用全域作为权重进行统计计算
        uniform_weights = torch.ones_like(self.hub) / (self.hub.shape[0] * self.hub.shape[1])
        features['global_skewness_u'] = self._weighted_skewness(u, uniform_weights)
        features['global_kurtosis_u'] = self._weighted_kurtosis(u, uniform_weights)
        
        # 极值风速比 (95分位数与5分位数的比值)
        speed_flat = speed.reshape(speed.shape[0], -1)
        speed_95 = torch.quantile(speed_flat, 0.95, dim=1)
        speed_05 = torch.quantile(speed_flat, 0.05, dim=1)
        features['extreme_wind_ratio'] = speed_95 / (speed_05 + 1e-8)
        
        # ===== Group 6: 新增能量分布特征 (3个) =====
        # 动能密度
        kinetic_energy = 0.5 * (u**2 + v**2 + w**2)
        features['kinetic_energy_density'] = torch.mean(kinetic_energy, dim=(1, 2))
        
        # 剪切生产率 (du/dy * u'v' 的估计)
        shear_rate = torch.mean(torch.abs(torch.diff(u, dim=1)), dim=(1, 2))
        features['shear_production_rate'] = shear_rate * torch.abs(features['global_reynolds_stress'])
        
        # 能量级联指标 (大小尺度湍流能量比)
        # 大尺度：低通滤波后的TKE
        u_smooth = torch.nn.functional.avg_pool2d(u.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        v_smooth = torch.nn.functional.avg_pool2d(v.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        w_smooth = torch.nn.functional.avg_pool2d(w.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        large_scale_tke = 0.5 * (torch.std(u_smooth, dim=(1,2))**2 + torch.std(v_smooth, dim=(1,2))**2 + torch.std(w_smooth, dim=(1,2))**2)
        features['energy_cascade_indicator'] = large_scale_tke / (features['global_tke'] + 1e-8)
        
        # 按顺序排列特征
        ordered_features = [features[name] for name in self.feature_names]
        return torch.stack(ordered_features, dim=1)


# --- 测试函数 ---
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
