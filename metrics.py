"""
metrics.py - Thư viện đánh giá dùng chung cho cả nhóm (RF, XGBoost, GraphSAGE, MLP).

Quy ước nhãn nhị phân: 0 = Normal (Benign), 1 = Attack (lớp dương).
Các chỉ số chính: F1-score, Recall, Confusion Matrix. Bổ sung: Precision, Macro-F1, FPR, FNR.

Cách dùng:
    import metrics
    res = metrics.evaluate_and_save("random_forest", "test", y_true, y_pred, y_score,
                                    label_multi=label_multi, out_dir="results")
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

NORMAL, ATTACK = 0, 1
CLASS_NAMES = ["Normal", "Attack"]


def compute_metrics(y_true, y_pred, y_score=None):
    """Chỉ số cho bài toán nhị phân Normal/Attack. Trả về dict (có thể ghi JSON)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[NORMAL, ATTACK])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    res = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=ATTACK, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=ATTACK, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=ATTACK, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[NORMAL, ATTACK], zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=[NORMAL, ATTACK], zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,   # Normal bị báo nhầm là Attack
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,   # Attack bị bỏ sót
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": cm.tolist(),                      # [[TN, FP], [FN, TP]]
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        res["roc_auc"] = float(roc_auc_score(y_true, y_score))
        res["pr_auc"] = float(average_precision_score(y_true, y_score))
    return res


def compute_multiclass_metrics(y_true, y_pred, labels=None):
    """Chỉ số đa lớp (dùng cho phân tích lỗi theo loại tấn công). Nhãn là chuỗi hoặc số."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = pd.Series(y_true).value_counts().reindex(labels).fillna(0).astype(int).to_numpy()
    per_class = pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f, "support": support})
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": list(labels),
    }


def detection_rate_by_type(label_multi, y_pred):
    """Với mỗi loại nhãn gốc: tỉ lệ bị dự đoán là Attack.
    Dòng Benign chính là False Positive Rate; các dòng tấn công chính là Recall theo từng loại."""
    d = pd.DataFrame({"label": np.asarray(label_multi), "pred_attack": np.asarray(y_pred).astype(int)})
    g = d.groupby("label")["pred_attack"].agg(n="size", predicted_attack="sum")
    g["rate_predicted_attack"] = g["predicted_attack"] / g["n"]
    return g.reset_index().sort_values("n", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", normalize=False, save_path=None, show=True):
    import matplotlib
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    class_names = class_names or CLASS_NAMES
    data = cm.astype(float)
    if normalize:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Dự đoán (Predicted)")
    ax.set_ylabel("Thực tế (Actual)")
    ax.set_title(title)
    thresh = data.max() / 2 if data.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{data[i, j]:.2%}" if normalize else f"{int(cm[i, j]):,}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if data[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def evaluate_and_save(model_name, split_name, y_true, y_pred, y_score=None, label_multi=None,
                      out_dir="results", verbose=True):
    """Tính chỉ số, in kết quả, lưu JSON + ảnh Confusion Matrix (+ CSV theo loại tấn công nếu có label_multi)."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{model_name}_{split_name}"
    res = compute_metrics(y_true, y_pred, y_score)
    res.update({"model": model_name, "split": split_name})
    with open(os.path.join(out_dir, f"{tag}_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    plot_confusion_matrix(res["confusion_matrix"], title=f"{model_name} - {split_name}",
                          save_path=os.path.join(out_dir, f"{tag}_confusion_matrix.png"), show=verbose)
    if label_multi is not None:
        detection_rate_by_type(label_multi, y_pred).to_csv(
            os.path.join(out_dir, f"{tag}_detection_by_type.csv"), index=False)
    if verbose:
        print(f"[{tag}] F1={res['f1']:.4f} | Recall={res['recall']:.4f} | Precision={res['precision']:.4f} | "
              f"Macro-F1={res['macro_f1']:.4f} | FPR={res['fpr']:.4f} | CM(TN,FP,FN,TP)="
              f"({res['tn']:,}, {res['fp']:,}, {res['fn']:,}, {res['tp']:,})")
    return res
