"""
Evaluation helpers: accuracy, macro-F1, AUC-ROC per class, and confusion matrices —
matching the "Evaluation Metrics" targets in the methodology slide
(overall accuracy ~88%, macro F1 ~0.82, AUC-ROC per class, confusion matrices).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@torch.no_grad()
def get_predictions(model, loader, device: str = "cpu"):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        spatial = batch["spatial"].to(device)
        weather = batch["weather"].to(device)
        logits = model(spatial, weather)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    return preds, probs, labels.numpy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, class_names: list[str]) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "report": classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
    }
    try:
        metrics["auc_roc_per_class"] = roc_auc_score(
            y_true, y_probs, multi_class="ovr", average=None, labels=list(range(len(class_names)))
        )
    except ValueError:
        # can happen with very small / degenerate synthetic samples (e.g. a class missing in y_true)
        metrics["auc_roc_per_class"] = None
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], save_path: str | None = None, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_training_curves(history: dict, save_path: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
