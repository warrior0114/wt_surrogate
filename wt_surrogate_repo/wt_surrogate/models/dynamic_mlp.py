"""Dynamic multi-head MLP model."""

import torch
import torch.nn as nn
from typing import Dict, List

"""Dynamic multi-head MLP model.

This module intentionally keeps backward-compatible constructor argument names to
support older training scripts.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

class DynamicMLPWindTurbineModel(nn.Module):
    """
    Dynamic multi-head MLP for wind turbine surrogate modeling.

    Backward-compatible constructor argument names:
    - Pass (feature_dim, operational_param_dim) OR (in_wind_dim, in_input_dim).
    - Pass dropout_rate OR dropout.
    - use_layernorm optionally inserts LayerNorm after Linear layers.
    """

    def __init__(
        self,
        feature_dim: Optional[int] = None,
        operational_param_dim: Optional[int] = None,
        task_config: Optional[Dict[str, int]] = None,
        hidden_dims: Optional[List[int]] = None,
        op_branch_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        *,
        in_wind_dim: Optional[int] = None,
        in_input_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        use_layernorm: bool = False,
    ):
        super().__init__()

        if task_config is None:
            raise ValueError("task_config must be provided, e.g. {'loads': 6} or {'power':1, 'loads':5}.")
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        if op_branch_dims is None:
            op_branch_dims = [64, 32]

        wind_dim = feature_dim if feature_dim is not None else in_wind_dim
        inp_dim = operational_param_dim if operational_param_dim is not None else in_input_dim
        if wind_dim is None or inp_dim is None:
            raise ValueError("Must provide feature_dim & operational_param_dim (preferred) or in_wind_dim & in_input_dim (alias).")

        dr = float(dropout_rate if dropout is None else dropout)

        # 1) operational/input branch
        op_layers = []
        d = int(inp_dim)
        for h in op_branch_dims:
            h = int(h)
            op_layers.append(nn.Linear(d, h))
            if use_layernorm:
                op_layers.append(nn.LayerNorm(h))
            op_layers.append(nn.LeakyReLU(0.01))
            if dr > 0:
                op_layers.append(nn.Dropout(dr))
            d = h
        self.operational_branch = nn.Sequential(*op_layers)

        # 2) shared trunk
        trunk_layers = []
        d = int(wind_dim) + int(op_branch_dims[-1])
        for h in hidden_dims:
            h = int(h)
            trunk_layers.append(nn.Linear(d, h))
            if use_layernorm:
                trunk_layers.append(nn.LayerNorm(h))
            trunk_layers.append(nn.LeakyReLU(0.01))
            if dr > 0:
                trunk_layers.append(nn.Dropout(dr))
            d = h
        self.mlp = nn.Sequential(*trunk_layers)

        # 3) heads
        final_dim = int(hidden_dims[-1])
        self.output_heads = nn.ModuleDict()
        self.task_names = list(task_config.keys())

        for task_name, out_dim in task_config.items():
            out_dim = int(out_dim)
            hid = 64 if out_dim > 1 else 32
            head = nn.Sequential(
                nn.Linear(final_dim, hid),
                nn.LeakyReLU(0.01),
                nn.Linear(hid, out_dim),
            )
            self.output_heads[task_name] = head

    def forward(self, features: torch.Tensor, operational_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        op_features = self.operational_branch(operational_params)
        x = torch.cat([features, op_features], dim=1)
        shared = self.mlp(x)
        return {name: self.output_heads[name](shared) for name in self.task_names}

# --- 示例和测试代码 ---
def test_dynamic_model():
    # 1. 定义一个任务配置
    # 假设我们这次想预测叶根载荷和偏航力矩
    my_task_config = {
        'root_loads': 2,       # 预测 RootFxb1, RootMyb1 (2维)
        'yaw_moments': 3     # 预测 YawBrMxp, YawBrMyp, YawBrMzp (3维)
    }

    # 2. 创建模型
    model = DynamicMLPWindTurbineModel(
        feature_dim=13, 
        operational_param_dim=10,
        task_config=my_task_config
    )
    print("\n动态模型已成功创建！结构如下：")
    print(model)
    model.eval()

    # 3. 模拟输入数据
    features_data = torch.randn(4, 13)
    operational_data = torch.randn(4, 10)

    # 4. 前向传播
    with torch.no_grad():
        outputs = model(features_data, operational_data)

    print("\n模型输出 (是一个字典):")
    # 检查输出是否符合预期
    for task_name, tensor in outputs.items():
        print(f"  - 任务 '{task_name}': 输出形状 {tensor.shape}")
        # 验证形状是否与配置匹配
        assert tensor.shape == (4, my_task_config[task_name])
    
    print("\n测试成功！模型能根据配置动态生成输出。")
    return model


# --- START OF FILE model_v2_attentive_mlp.py ---

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class AttentiveMLPWindTurbineModel(nn.Module):
    """
    带有嵌入式特征注意力的动态多头风力机MLP模型。
    - 在进入主干网络前，增加一个可学习的注意力层来对输入特征进行加权。
    - 能够直接输出学习到的特征重要性得分。
    """
    def __init__(self, 
                 feature_dim: int, 
                 operational_param_dim: int, 
                 task_config: Dict[str, int],
                 hidden_dims: List[int] = [512, 256, 128], 
                 op_branch_dims: List[int] = [64, 32], 
                 dropout_rate: float = 0.2):
        super().__init__()
        
        # ==================== 核心改动 1: 特征注意力层 ====================
        # 创建一个与输入特征维度相同的可学习参数（权重）
        self.feature_attention_logits = nn.Parameter(torch.ones(feature_dim))
        # 使用Softmax确保所有权重加起来为1，便于解释
        self.feature_attention_activation = nn.Softmax(dim=-1)
        print(f"--- 模型初始化: 添加了嵌入式特征注意力层 (维度: {feature_dim}) ---")
        # =================================================================

        # 1. 工况参数分支 (与原版相同)
        op_layers = []
        input_op_dim = operational_param_dim
        for hidden_dim in op_branch_dims:
            op_layers.append(nn.Linear(input_op_dim, hidden_dim))
            op_layers.append(nn.LeakyReLU(0.01))
            input_op_dim = hidden_dim
        self.operational_branch = nn.Sequential(*op_layers)
        
        # 2. 共享主干MLP (与原版相同)
        mlp_layers = []
        # 注意：这里的输入维度仍然是 feature_dim，因为注意力层不改变维度
        input_mlp_dim = feature_dim + op_branch_dims[-1]
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_mlp_dim, hidden_dim))
            mlp_layers.append(nn.LeakyReLU(0.01))
            mlp_layers.append(nn.Dropout(dropout_rate))
            input_mlp_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # 3. 动态创建输出头 (与原版相同)
        final_shared_dim = hidden_dims[-1]
        self.output_heads = nn.ModuleDict()
        self.task_names = list(task_config.keys())
        
        for task_name, output_dim in task_config.items():
            head = nn.Sequential(
                nn.Linear(final_shared_dim, 64 if output_dim > 1 else 32),
                nn.LeakyReLU(0.01),
                nn.Linear(64 if output_dim > 1 else 32, output_dim)
            )
            self.output_heads[task_name] = head

    def get_feature_importance(self) -> torch.Tensor:
        """
        辅助函数，用于在训练后获取特征重要性得分。
        """
        return self.feature_attention_activation(self.feature_attention_logits)

    def forward(self, features: torch.Tensor, operational_params: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        前向传播。
        
        Returns:
            - predictions (Dict): 预测值的字典
            - attention_weights (Tensor): 学习到的特征权重
        """
        # ==================== 核心改动 2: 应用注意力权重 ====================
        # 计算特征注意力权重
        attention_weights = self.feature_attention_activation(self.feature_attention_logits)
        
        # 将权重应用到输入特征上
        # features: [batch_size, feature_dim]
        # attention_weights: [feature_dim] -> PyTorch会自动广播
        features_att = features * attention_weights
        # =================================================================

        # 后续流程与原版相同，但使用加权后的 features_att
        op_features = self.operational_branch(operational_params)
        x = torch.cat([features_att, op_features], dim=1)
        shared_features = self.mlp(x)
        
        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.output_heads[task_name](shared_features)
            
        return outputs, attention_weights

# --- 测试函数 ---
def test_attentive_mlp_model():
    my_task_config = {'loads': 6}
    FEATURE_DIM = 45 # 真实特征维度
    
    model = AttentiveMLPWindTurbineModel(
        feature_dim=FEATURE_DIM, 
        operational_param_dim=6,
        task_config=my_task_config
    )
    
    features_data = torch.randn(4, FEATURE_DIM)
    op_data = torch.randn(4, 6)
    
    # forward现在返回两个值
    predictions, weights = model(features_data, op_data)
    
    print("\n模型输出形状:", predictions['loads'].shape)
    assert predictions['loads'].shape == (4, 6)
    
    print("注意力权重形状:", weights.shape)
    assert weights.shape == (FEATURE_DIM,)
    
    importance = model.get_feature_importance()
    print("提取的重要性得分总和 (应约等于1):", importance.sum().item())
    assert torch.isclose(importance.sum(), torch.tensor(1.0))
    print("\n测试成功！")

if __name__ == "__main__":
    test_attentive_mlp_model()

