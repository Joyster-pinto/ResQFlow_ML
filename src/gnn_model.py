"""
ResQFlow ML — Graph Neural Network Model
GraphSAGE-based travel time predictor that understands road network topology.

Each SUMO edge becomes a GNN node. Edges in the GNN connect sequential road
segments (where one SUMO edge feeds into the next). This lets the network
"see" that congestion on one road propagates to adjacent roads.
"""

import logging

logger = logging.getLogger(__name__)

# ── Torch optional import ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    try:
        from torch_geometric.nn import SAGEConv, BatchNorm
        _TORCH_GEO = True
    except (ImportError, OSError):
        _TORCH_GEO = False
        logger.warning("torch_geometric not available. GNN model disabled.")
    _TORCH = True
except (ImportError, OSError) as _e:
    # OSError covers Windows WinError 1114 (DLL init failure for c10.dll)
    _TORCH = False
    _TORCH_GEO = False
    logger.warning(f"PyTorch not available ({type(_e).__name__}: {_e}). "
                   "GNN model disabled; using LightGBM + CatBoost only.")


def gnn_available() -> bool:
    """Return True if both torch and torch_geometric are installed."""
    return _TORCH and _TORCH_GEO


# ── Model definition ──────────────────────────────────────────────────────────

if gnn_available():
    class RoadGNN(nn.Module):
        """
        2-layer GraphSAGE for road edge travel-time prediction.

        Architecture:
          x (N, in_features)
            → SAGEConv(in, hidden) + BN + ReLU + Dropout
            → SAGEConv(hidden, hidden) + BN + ReLU
            → MLP(hidden → 64 → 1)  [travel time in seconds]

        The SAGEConv layers aggregate features from neighboring edges so the
        model accounts for congestion propagation across the road network.
        """

        def __init__(self, in_features: int = 12, hidden: int = 128, dropout: float = 0.2):
            super().__init__()
            self.conv1   = SAGEConv(in_features, hidden)
            self.bn1     = BatchNorm(hidden)
            self.conv2   = SAGEConv(hidden, hidden)
            self.bn2     = BatchNorm(hidden)
            self.dropout = nn.Dropout(dropout)
            self.mlp     = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x, edge_index):
            # Layer 1
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout(x)
            # Layer 2
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            # Regression head
            return self.mlp(x).squeeze(-1)      # shape: (N,)

else:
    # Stub so imports don't break when torch is absent
    class RoadGNN:                              # type: ignore
        def __init__(self, *a, **kw):
            raise RuntimeError("PyTorch / torch_geometric not installed. "
                               "Run: pip install torch torch_geometric")
