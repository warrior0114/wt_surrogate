"""Dynamic multi-head MLP model."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

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

# --- Example and Test Code ---
def test_dynamic_model():
    # 1. Define a task configuration
    # Suppose we want to predict root loads and yaw moments
    my_task_config = {
        'root_loads': 2,       # Predict RootFxb1, RootMyb1 (2 dimensions)
        'yaw_moments': 3       # Predict YawBrMxp, YawBrMyp, YawBrMzp (3 dimensions)
    }

    # 2. Create model
    model = DynamicMLPWindTurbineModel(
        feature_dim=13, 
        operational_param_dim=10,
        task_config=my_task_config
    )
    print("\nDynamic model created successfully! Structure:")
    print(model)
    model.eval()

    # 3. Simulate input data
    features_data = torch.randn(4, 13)
    operational_data = torch.randn(4, 10)

    # 4. Forward pass
    with torch.no_grad():
        outputs = model(features_data, operational_data)

    print("\nModel Output (is a dictionary):")
    # Check if output meets expectations
    for task_name, tensor in outputs.items():
        print(f"  - Task '{task_name}': Output shape {tensor.shape}")
        # Verify shape matches configuration
        assert tensor.shape == (4, my_task_config[task_name])
    
    print("\nTest successful! Model generates output dynamically based on configuration.")
    return model


# --- START OF FILE model_v2_attentive_mlp.py ---

class AttentiveMLPWindTurbineModel(nn.Module):
    """
    Dynamic multi-head MLP model for wind turbines with embedded feature attention.
    - Adds a learnable attention layer to weight input features before entering the backbone.
    - Capable of directly outputting learned feature importance scores.
    """
    def __init__(self, 
                 feature_dim: int, 
                 operational_param_dim: int, 
                 task_config: Dict[str, int],
                 hidden_dims: List[int] = [512, 256, 128], 
                 op_branch_dims: List[int] = [64, 32], 
                 dropout_rate: float = 0.2):
        super().__init__()
        
        # ==================== Core Change 1: Feature Attention Layer ====================
        # Create a learnable parameter (weight) with the same dimension as input features
        self.feature_attention_logits = nn.Parameter(torch.ones(feature_dim))
        # Use Softmax to ensure all weights sum to 1 for interpretability
        self.feature_attention_activation = nn.Softmax(dim=-1)
        print(f"--- Model Initialization: Added embedded feature attention layer (dim: {feature_dim}) ---")
        # =================================================================

        # 1. Operational parameter branch (same as original)
        op_layers = []
        input_op_dim = operational_param_dim
        for hidden_dim in op_branch_dims:
            op_layers.append(nn.Linear(input_op_dim, hidden_dim))
            op_layers.append(nn.LeakyReLU(0.01))
            input_op_dim = hidden_dim
        self.operational_branch = nn.Sequential(*op_layers)
        
        # 2. Shared Backbone MLP (same as original)
        mlp_layers = []
        # Note: Input dim here is still feature_dim because the attention layer does not change dimensions
        input_mlp_dim = feature_dim + op_branch_dims[-1]
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_mlp_dim, hidden_dim))
            mlp_layers.append(nn.LeakyReLU(0.01))
            mlp_layers.append(nn.Dropout(dropout_rate))
            input_mlp_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # 3. Dynamically create output heads (same as original)
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
        Helper function to get feature importance scores after training.
        """
        return self.feature_attention_activation(self.feature_attention_logits)

    def forward(self, features: torch.Tensor, operational_params: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            - predictions (Dict): Dictionary of predicted values
            - attention_weights (Tensor): Learned feature weights
        """
        # ==================== Core Change 2: Apply Attention Weights ====================
        # Calculate feature attention weights
        attention_weights = self.feature_attention_activation(self.feature_attention_logits)
        
        # Apply weights to input features
        # features: [batch_size, feature_dim]
        # attention_weights: [feature_dim] -> PyTorch will automatically broadcast
        features_att = features * attention_weights
        # =================================================================

        # Subsequent process is the same as original, but uses weighted features_att
        op_features = self.operational_branch(operational_params)
        x = torch.cat([features_att, op_features], dim=1)
        shared_features = self.mlp(x)
        
        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.output_heads[task_name](shared_features)
            
        return outputs, attention_weights

# --- Test Function ---
def test_attentive_mlp_model():
    my_task_config = {'loads': 6}
    FEATURE_DIM = 45 # Actual feature dimension
    
    model = AttentiveMLPWindTurbineModel(
        feature_dim=FEATURE_DIM, 
        operational_param_dim=6,
        task_config=my_task_config
    )
    
    features_data = torch.randn(4, FEATURE_DIM)
    op_data = torch.randn(4, 6)
    
    # forward now returns two values
    predictions, weights = model(features_data, op_data)
    
    print("\nModel output shape:", predictions['loads'].shape)
    assert predictions['loads'].shape == (4, 6)
    
    print("Attention weights shape:", weights.shape)
    assert weights.shape == (FEATURE_DIM,)
    
    importance = model.get_feature_importance()
    print("Sum of extracted importance scores (should be approx 1):", importance.sum().item())
    assert torch.isclose(importance.sum(), torch.tensor(1.0))
    print("\nTest successful!")

if __name__ == "__main__":
    test_attentive_mlp_model()