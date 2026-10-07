"""
Training loop for the Hybrid CNN-LSTM wildfire risk classifier, plus the baseline
comparisons called for in the methodology (Logistic Regression, Random Forest, XGBoost).

Usage:
    python -m src.training.train --synthetic --epochs 20 --horizon 1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.synthetic_data import AlaskaWildfireSyntheticDataset, spatial_block_split
from src.models.cnn_lstm import build_model, FocalLoss


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataloaders(cfg: dict, horizon: int):
    dataset = AlaskaWildfireSyntheticDataset(cfg, horizon=horizon, seed=cfg["project"]["seed"])
    train_idx, val_idx, test_idx = spatial_block_split(dataset, cfg, seed=cfg["project"]["seed"])

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
    return dataset, train_loader, val_loader, test_loader


def class_weights_from_labels(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            spatial = batch["spatial"].to(device)
            weather = batch["weather"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad()

            logits = model(spatial, weather)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_total += labels.size(0)

    return total_loss / n_total, n_correct / n_total


def train_cnn_lstm(cfg: dict, horizon: int, device: str = "cpu"):
    set_seed(cfg["project"]["seed"])
    dataset, train_loader, val_loader, test_loader = get_dataloaders(cfg, horizon)

    model = build_model(cfg, dataset.n_spatial_channels, cfg["data"]["n_weather_features"]).to(device)

    weights = None
    if cfg["training"]["oversample_minority"]:
        train_labels = np.array([dataset.labels[i] for i in train_loader.dataset.indices])
        weights = class_weights_from_labels(train_labels, cfg["model"]["num_classes"]).to(device)

    if cfg["training"]["use_focal_loss"]:
        criterion = FocalLoss(gamma=cfg["training"]["focal_gamma"], weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_val_loss = float("inf")
    patience = cfg["training"]["early_stopping_patience"]
    patience_counter = 0
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["training"]["checkpoint_dir"], f"cnn_lstm_horizon{horizon}.pt")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    pbar = tqdm(range(cfg["training"]["epochs"]), desc=f"Training (horizon={horizon}mo)")
    for epoch in pbar:
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        pbar.set_postfix(train_loss=f"{train_loss:.3f}", val_loss=f"{val_loss:.3f}", val_acc=f"{val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, history, (dataset, train_loader, val_loader, test_loader)


def train_baselines(dataset, train_loader, test_loader, cfg: dict):
    """Flattened-feature Logistic Regression / Random Forest / XGBoost baselines,
    for comparison against the CNN-LSTM per the evaluation methodology."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    def flatten_split(loader):
        X, y = [], []
        for batch in loader:
            spatial = batch["spatial"].numpy().reshape(batch["spatial"].shape[0], -1)
            weather = batch["weather"].numpy().reshape(batch["weather"].shape[0], -1)
            X.append(np.concatenate([spatial, weather], axis=1))
            y.append(batch["label"].numpy())
        return np.concatenate(X), np.concatenate(y)

    X_train, y_train = flatten_split(train_loader)
    X_test, y_test = flatten_split(test_loader)

    results = {}
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=cfg["project"]["seed"]),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, eval_metric="mlogloss",
            random_state=cfg["project"]["seed"],
        ),
    }
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = (preds == y_test).mean()
        results[name] = {"model": clf, "test_acc": acc, "preds": preds, "y_test": y_test}
        print(f"{name}: test accuracy = {acc:.3f}")

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--horizon", type=int, default=1, choices=[1, 3, 6])
    p.add_argument("--synthetic", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    model, history, loaders = train_cnn_lstm(cfg, horizon=args.horizon, device=args.device)
    dataset, train_loader, val_loader, test_loader = loaders

    test_loss, test_acc = run_epoch(
        model, test_loader,
        FocalLoss(gamma=cfg["training"]["focal_gamma"]) if cfg["training"]["use_focal_loss"]
        else torch.nn.CrossEntropyLoss(),
        optimizer=None, device=args.device, train=False,
    )
    print(f"\nCNN-LSTM test accuracy (horizon={args.horizon}mo): {test_acc:.3f}")

    print("\nTraining baselines for comparison...")
    train_baselines(dataset, train_loader, test_loader, cfg)


if __name__ == "__main__":
    main()
