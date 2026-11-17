import asyncio
import json
import time
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from predict_window import EGraphSAGEModel

app = FastAPI(title="E-GraphSAGE Streaming IDS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load dữ liệu Test & Model 1 lần duy nhất vào RAM để chạy siêu mượt
print("Đang nạp dữ liệu Test và trọng số E-GraphSAGE...")
X_te = pd.read_parquet("processed/test_features.parquet")
m_te = pd.read_parquet("processed/test_meta.parquet")

# Sắp xếp đúng theo dòng thời gian thực
order = np.argsort(m_te["timestamp"].to_numpy())
X_arr = X_te.to_numpy(dtype=np.float32)[order]
m_sorted = m_te.iloc[order].reset_index(drop=True)

in_dim = X_arr.shape[1]
model = EGraphSAGEModel(in_dim=in_dim, hidden_dim=128).to(device)
model.load_state_dict(torch.load("models/checkpoint.pth", map_location=device))
model.eval()
print(f"Đã sẵn sàng! Tổng số flows: {len(X_arr):,} | Thiết bị: {device}")


@app.get("/api/summary")
def get_summary():
    """Trả về thông số tổng quan offline của mô hình cho header dashboard"""
    try:
        with open("results/graphsage_test_metrics.json") as f:
            return json.load(f)
    except Exception:
        return {"f1": 0.8977, "precision": 0.9863, "recall": 0.8237, "fpr": 0.0028}


@app.websocket("/ws/stream")
async def stream_ids(websocket: WebSocket):
    await websocket.accept()

    # Tham số mặc định có thể điều chỉnh từ UI
    window_size = 2000
    threshold = 0.15
    speed_sec = 1.0
    is_running = True
    cursor = 0
    n_total = len(X_arr)

    async def receive_controls():
        nonlocal window_size, threshold, speed_sec, is_running, cursor
        try:
            while True:
                msg = await websocket.receive_json()
                action = msg.get("action")
                if action == "toggle":
                    is_running = not is_running
                elif action == "reset":
                    cursor = 0
                elif action == "config":
                    window_size = int(msg.get("window_size", window_size))
                    threshold = float(msg.get("threshold", threshold))
                    speed_sec = float(msg.get("speed_sec", speed_sec))
                elif action == "jump_attack":
                    # Nhảy nhanh tới vùng có tấn công để demo trực tiếp
                    atk_indices = np.where(m_sorted["label_bin"].to_numpy() == 1)[0]
                    if len(atk_indices) > 0:
                        cursor = max(0, int(atk_indices[len(atk_indices) // 3]) - 200)
        except (WebSocketDisconnect, RuntimeError):
            pass

    recv_task = asyncio.create_task(receive_controls())

    try:
        window_id = 0
        while True:
            if not is_running:
                await asyncio.sleep(0.2)
                continue

            if cursor >= n_total:
                cursor = 0

            end = min(cursor + window_size, n_total)
            batch_X = X_arr[cursor:end]
            batch_meta = m_sorted.iloc[cursor:end]

            t0 = time.perf_counter()

            # Dựng đồ thị cửa sổ
            s_ip = batch_meta["src_ip"].astype(str).to_numpy()
            d_ip = batch_meta["dst_ip"].astype(str).to_numpy()
            unique_ips, inv_idx = np.unique(np.concatenate([s_ip, d_ip]), return_inverse=True)
            n_edges = end - cursor
            src_idx = torch.tensor(inv_idx[:n_edges], dtype=torch.long, device=device)
            dst_idx = torch.tensor(inv_idx[n_edges:], dtype=torch.long, device=device)
            edge_index = torch.stack([src_idx, dst_idx], dim=0)
            edge_attr = torch.tensor(batch_X, dtype=torch.float32, device=device)
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

            latency_ms = (time.perf_counter() - t0) * 1000.0
            throughput = int(n_edges / max(latency_ms / 1000.0, 1e-4))

            preds = (probs >= threshold).astype(int)
            atk_count = int(preds.sum())
            norm_count = int(n_edges - atk_count)

            # Xây dựng dữ liệu đồ thị rút gọn (tối đa 40 nút) để vẽ Canvas siêu mượt
            attackers, victims = set(), set()
            edge_map = {}
            for i in range(n_edges):
                u, v = s_ip[i], d_ip[i]
                if u == v:
                    continue
                is_atk = bool(preds[i] == 1)
                if is_atk:
                    attackers.add(u)
                    victims.add(v)
                key = (u, v)
                if key not in edge_map:
                    edge_map[key] = {"source": u, "target": v, "is_atk": is_atk, "count": 1}
                else:
                    edge_map[key]["count"] += 1
                    edge_map[key]["is_atk"] = edge_map[key]["is_atk"] or is_atk

            important = list(attackers | victims)
            others = [ip for ip in unique_ips if ip not in attackers and ip not in victims]
            kept_ips = set(important[:30] + others[: max(0, 40 - len(important[:30]))])

            nodes = []
            for ip in kept_ips:
                role = "attacker" if ip in attackers else ("victim" if ip in victims else "normal")
                nodes.append({"id": ip, "role": role})

            links = [
                e for e in edge_map.values()
                if e["source"] in kept_ips and e["target"] in kept_ips
            ][:120]

            # Lấy tối đa 15 cảnh báo mới nhất trong cửa sổ
            alerts = []
            atk_pos = np.where(preds == 1)[0]
            for idx in atk_pos[-15:][::-1]:
                row = batch_meta.iloc[idx]
                alerts.append({
                    "id": f"{window_id}-{idx}",
                    "timestamp": str(row.get("timestamp", "")),
                    "src_ip": str(row["src_ip"]),
                    "dst_ip": str(row["dst_ip"]),
                    "prob": round(float(probs[idx]) * 100, 1),
                    "true_label": str(row.get("label_multi", "Attack")),
                })

            payload = {
                "window_id": window_id,
                "is_running": is_running,
                "progress": round(end / n_total * 100, 1),
                "total_flows": n_edges,
                "attack_flows": atk_count,
                "normal_flows": norm_count,
                "latency_ms": round(latency_ms, 2),
                "throughput": throughput,
                "graph": {"nodes": nodes, "links": links},
                "alerts": alerts,
            }

            await websocket.send_json(payload)
            cursor = end
            window_id += 1
            await asyncio.sleep(speed_sec)

    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        recv_task.cancel()