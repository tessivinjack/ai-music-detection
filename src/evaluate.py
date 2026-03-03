import argparse
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.config import ensure_dirs
from src.dataset import MelNpyDataset
from src.train_baseline import SimpleCNN  # reuse exact architecture

@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    ys = []
    probs = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(fake)
        probs.extend(p.tolist())
        ys.extend(y.numpy().tolist())
    return np.array(ys), np.array(probs)

def pick_thresholds_from_pr(prec, rec, thr, targets=(0.90, 0.95, 0.99)):
    """
    precision_recall_curve returns prec, rec of length N+1, and thr of length N.
    Threshold thr[i] corresponds to prec[i+1], rec[i+1] (scikit-learn convention).
    We'll align them so each threshold has a matching precision/recall.
    """
    if len(thr) == 0:
        return []

    prec_t = prec[1:]
    rec_t = rec[1:]

    rows = []

    # Default threshold = 0.5 row (computed by nearest threshold)
    default_t = 0.5
    j = int(np.argmin(np.abs(thr - default_t)))
    rows.append({
        "policy": "default_0.50",
        "threshold": float(thr[j]),
        "precision": float(prec_t[j]),
        "recall": float(rec_t[j]),
    })

    # High precision policies
    for target_p in targets:
        idxs = np.where(prec_t >= target_p)[0]
        if len(idxs) == 0:
            rows.append({
                "policy": f"precision_ge_{target_p:.2f}",
                "threshold": None,
                "precision": None,
                "recall": None,
                "note": "Not achievable on this eval set"
            })
            continue

        # choose the highest recall among thresholds that meet precision target
        best = idxs[np.argmax(rec_t[idxs])]
        rows.append({
            "policy": f"precision_ge_{target_p:.2f}",
            "threshold": float(thr[best]),
            "precision": float(prec_t[best]),
            "recall": float(rec_t[best]),
        })

    # High recall policy (example: recall >= 0.95)
    target_r = 0.95
    idxs = np.where(rec_t >= target_r)[0]
    if len(idxs) == 0:
        rows.append({
            "policy": f"recall_ge_{target_r:.2f}",
            "threshold": None,
            "precision": None,
            "recall": None,
            "note": "Not achievable on this eval set"
        })
    else:
        # choose the highest precision among thresholds that meet recall target
        best = idxs[np.argmax(prec_t[idxs])]
        rows.append({
            "policy": f"recall_ge_{target_r:.2f}",
            "threshold": float(thr[best]),
            "precision": float(prec_t[best]),
            "recall": float(rec_t[best]),
        })

    return rows

def main():
    parser = argparse.ArgumentParser(description="Evaluate the baseline CNN and write metrics/plots.")
    parser.add_argument("--val_csv", type=str, default="data/interim/splits/val.csv")
    parser.add_argument("--model_path", type=str, default="results/baseline_cnn.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    paths = ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MelNpyDataset(args.val_csv)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    y_true, p_fake = get_probs(model, loader, device)
    y_pred = (p_fake >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3)

    ap = average_precision_score(y_true, p_fake)
    prec, rec, thr = precision_recall_curve(y_true, p_fake)

    threshold_table = pick_thresholds_from_pr(prec, rec, thr, targets=(0.90, 0.95, 0.99))


    # Save metrics
    out = {
        "accuracy": float(acc),
        "average_precision": float(ap),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "threshold_policies": threshold_table,
    }
    metrics_path = paths.results / "eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Accuracy: {acc:.3f}")
    print(f"Avg Precision (PR-AUC): {ap:.3f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    print(f"\nSaved metrics: {metrics_path}")
    print("\nThreshold policy table:")
    for row in threshold_table:
        print(row)

    # Plot confusion matrix
    fig_dir = paths.results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (threshold=0.5)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["real", "fake"])
    plt.yticks([0, 1], ["real", "fake"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    cm_path = fig_dir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {cm_path}")

    # Plot precision-recall curve
    plt.figure()
    plt.plot(rec, prec)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    pr_path = fig_dir / "precision_recall_curve.png"
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pr_path}")

if __name__ == "__main__":
    main()
