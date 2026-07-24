"""Feed-forward network used by the DINOv2 and VGGT transformer blocks."""

from __future__ import annotations

from typing import Optional

import mlx.nn as nn


class Mlp(nn.Module):
    """PyTorch-compatible two-layer GELU MLP.

    Attribute names intentionally match upstream VGGT checkpoint keys.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU(approx="none")
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)
