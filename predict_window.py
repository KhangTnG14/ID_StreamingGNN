import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class EGraphSAGEConv(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__()
        self.sage = SAGEConv(node_dim + edge_dim, out_dim, aggr="max")
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg_edge = torch.zeros((x.size(0), edge_attr.size(1)), device=x.device)
        agg_edge.index_add_(0, col, edge_attr)
        agg_edge.index_add_(0, row, edge_attr)
        
        deg = (torch.bincount(col, minlength=x.size(0)) + torch.bincount(row, minlength=x.size(0))).clamp(min=1).unsqueeze(1).float()
        agg_edge = agg_edge / deg
        
        h = torch.cat([x, agg_edge], dim=-1)
        return F.leaky_relu(self.norm(self.sage(h, edge_index)), negative_slope=0.1)

class EGraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = EGraphSAGEConv(in_dim, in_dim, hidden_dim)
        self.conv2 = EGraphSAGEConv(hidden_dim, in_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = torch.clamp(x, -10.0, 10.0)
        edge_attr = torch.clamp(edge_attr, -10.0, 10.0)
        
        h = self.conv1(x, edge_index, edge_attr)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr)
        
        src, dst = edge_index
        e_emb = self.edge_encoder(edge_attr)
        edge_rep = torch.cat([h[src], h[dst], e_emb], dim=-1)
        return self.classifier(edge_rep).squeeze(-1)

_CACHE = {}

def predict_window(
    df_batch: pd.DataFrame,
    checkpoint_path: str = "models/checkpoint.pth",
    preprocessor_path: str = "processed/preprocessor.joblib",
    threshold: float = 0.15
):
    """
    Hàm suy luận Streaming cho 1 cửa sổ/batch dữ liệu mạng (DataFrame).
    Trả về:
        - preds: mảng numpy 0 (Normal) / 1 (Attack)
        - probs: mảng numpy xác suất tấn công tương ứng từng dòng
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if "prep" not in _CACHE:
        _CACHE["prep"] = joblib.load(preprocessor_path)
    prep = _CACHE["prep"]
    
    feature_cols = prep["feature_cols"] if isinstance(prep, dict) else prep.feature_names_in_
    scaler = prep["scaler"] if isinstance(prep, dict) else prep
    
    X_raw = df_batch[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    in_dim = X_scaled.shape[1]
    
    if "model" not in _CACHE:
        model = EGraphSAGEModel(in_dim=in_dim, hidden_dim=128).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        _CACHE["model"] = model
    model = _CACHE["model"]
    
    src_col = "src_ip" if "src_ip" in df_batch.columns else "Source IP"
    dst_col = "dst_ip" if "dst_ip" in df_batch.columns else "Destination IP"
    s_ip = df_batch[src_col].astype(str).to_numpy()
    d_ip = df_batch[dst_col].astype(str).to_numpy()
    
    unique_ips, inv_idx = np.unique(np.concatenate([s_ip, d_ip]), return_inverse=True)
    n_edges = len(df_batch)
    src_idx = torch.tensor(inv_idx[:n_edges], dtype=torch.long, device=device)
    dst_idx = torch.tensor(inv_idx[n_edges:], dtype=torch.long, device=device)
    
    edge_index = torch.stack([src_idx, dst_idx], dim=0)
    edge_attr = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    num_nodes = len(unique_ips)
    
    node_sum = torch.zeros((num_nodes, in_dim), dtype=torch.float32, device=device)
    node_cnt = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
    node_sum.index_add_(0, src_idx, edge_attr)
    node_sum.index_add_(0, dst_idx, edge_attr)
    node_cnt.index_add_(0, src_idx, torch.ones((n_edges, 1), device=device))
    node_cnt.index_add_(0, dst_idx, torch.ones((n_edges, 1), device=device))
    x = node_sum / node_cnt.clamp(min=1.0)
    
    with torch.no_grad():
        probs = torch.sigmoid(model(x, edge_index, edge_attr)).cpu().numpy()
        
    preds = (probs >= threshold).astype(int)
    return preds, probs
