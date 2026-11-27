import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter


class EGraphSAGEConv(nn.Module):
    def __init__(self, node_in_dim, edge_dim, out_dim):
        super().__init__()
        self.sage = SAGEConv(node_in_dim + edge_dim, out_dim, aggr="max")
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        num_nodes = x.size(0)
        edge_ends = torch.cat([src, dst], dim=0)
        edge_feats = torch.cat([edge_attr, edge_attr], dim=0)
        agg_edge = scatter(edge_feats, edge_ends, dim=0, dim_size=num_nodes, reduce="mean")
        x_aug = torch.cat([x, agg_edge], dim=-1)
        out = self.sage(x_aug, edge_index)
        return self.act(self.bn(out))


class EGraphSAGEModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.conv1 = EGraphSAGEConv(node_dim, edge_dim, hidden_dim)
        self.conv2 = EGraphSAGEConv(hidden_dim, edge_dim, hidden_dim)

        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x = torch.clamp(data.x, -25.0, 25.0)
        edge_attr = torch.clamp(data.edge_attr, -25.0, 25.0)
        edge_index = data.edge_index

        h = self.conv1(x, edge_index, edge_attr)
        h = self.conv2(h, edge_index, edge_attr)

        src, dst = edge_index
        e_emb = self.edge_enc(edge_attr)

        out = torch.cat([h[src], h[dst], e_emb], dim=-1)
        return self.classifier(out).squeeze(-1)


class StreamingIDSPredictor:
    def __init__(self, bundle_dir="."):
        with open(f"{bundle_dir}/metadata.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.feature_cols = self.meta["feature_cols"]
        self.threshold = self.meta["best_threshold"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = joblib.load(f"{bundle_dir}/scaler.joblib")
        self.medians = joblib.load(f"{bundle_dir}/medians.joblib")

        self.model = EGraphSAGEModel(
            node_dim=self.meta["node_dim"],
            edge_dim=self.meta["edge_dim"],
            hidden_dim=self.meta["hidden_dim"],
            dropout=self.meta["dropout"],
        ).to(self.device)
        state = torch.load(f"{bundle_dir}/best_egraphsage.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_window(self, df_window: pd.DataFrame) -> pd.DataFrame:
        df = df_window.copy()
        df.columns = [c.strip() for c in df.columns]

        X = df[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(self.medians).astype(np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)

        ips = pd.Index(pd.concat([df["Source IP"], df["Destination IP"]]).unique())
        num_nodes = len(ips)
        src_t = torch.tensor(ips.get_indexer(df["Source IP"]), dtype=torch.long)
        dst_t = torch.tensor(ips.get_indexer(df["Destination IP"]), dtype=torch.long)
        edge_index = torch.stack([src_t, dst_t], dim=0)
        edge_attr = torch.tensor(X_scaled, dtype=torch.float32)

        edge_ends = torch.cat([src_t, dst_t], dim=0)
        edge_feats_dup = torch.cat([edge_attr, edge_attr], dim=0)
        node_mean = scatter(edge_feats_dup, edge_ends, dim=0, dim_size=num_nodes, reduce="mean")
        node_max = scatter(edge_feats_dup, edge_ends, dim=0, dim_size=num_nodes, reduce="max")
        node_x = torch.cat([node_mean, node_max], dim=-1)

        data = Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(data)).cpu().numpy()

        df["attack_prob"] = probs
        df["pred_label"] = (probs >= self.threshold).astype(int)
        return df
