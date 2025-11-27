import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .custom_loss import CustomLoss

# -------------------
# Activation function
# -------------------
def get_activation(name: Optional[str]) -> nn.Module:
    if name is None:
        return nn.Identity()
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "custom":
        return CustomLoss()  
    # default: identity if unknown
    return nn.Identity()


# ------------
# Head modules
# ------------
class LinearHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, output_activation: Optional[str]):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = get_activation(output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return x


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        hidden_activation: Optional[str],
        output_activation: Optional[str],
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_activation(hidden_activation))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(get_activation(output_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------
# Head "factory"
# ----------------
def init_head(head_cfg: Dict[str, Any], input_dim: int) -> nn.Module:
    head_type = head_cfg.get("type", "linear").lower()
    output_dim = head_cfg["output_dim"]

    if head_type == "linear":
        return LinearHead(
            input_dim=input_dim,
            output_dim=output_dim,
            output_activation=head_cfg.get("output_activation"),
        )

    elif head_type == "mlp":
        return MLPHead(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=head_cfg["hidden_dim"],
            num_layers=head_cfg.get("num_layers", 1),
            dropout=head_cfg.get("dropout", 0.0),
            hidden_activation=head_cfg.get("hidden_activation", "relu"),
            output_activation=head_cfg.get("output_activation"),
        )

    else:
        raise ValueError(f"Unsupported head type: {head_type}")

