"""Multi-task training: regression on normalized returns + BCE on cumulative direction."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from forex_signal.data.features import FEATURE_COLUMNS, WindowedDataset
from forex_signal.model.lnn import ForexLNN

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    units: int = 48
    dropout: float = 0.1
    batch_size: int = 128
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 6
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    device: str = "cpu"
    seed: int = 42
    direction_loss_weight: float = 1.0  # weight on BCE relative to MSE


@dataclass
class TrainResult:
    best_val_loss: float
    test_loss: float
    test_directional_accuracy: float
    test_classifier_accuracy: float
    test_classifier_p70_acc: float    # accuracy on samples where |p-0.5| >= 0.2 (confident)
    test_classifier_p70_count: int
    epochs_run: int
    history: list[dict]
    model_path: str


def _split(ds: WindowedDataset, val_frac: float, test_frac: float):
    n = len(ds.X)
    test_n = int(n * test_frac)
    val_n = int(n * val_frac)
    train_n = n - val_n - test_n
    X_tr, y_tr = ds.X[:train_n], ds.y[:train_n]
    X_val, y_val = ds.X[train_n : train_n + val_n], ds.y[train_n : train_n + val_n]
    X_te, y_te = ds.X[train_n + val_n :], ds.y[train_n + val_n :]
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def _directional_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    pred_dir = np.sign(pred.sum(axis=1))
    targ_dir = np.sign(target.sum(axis=1))
    mask = targ_dir != 0
    if mask.sum() == 0:
        return 0.0
    return float((pred_dir[mask] == targ_dir[mask]).mean())


def train(
    dataset: WindowedDataset,
    config: TrainConfig,
    save_path: Path | str,
) -> TrainResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = _split(
        dataset, config.val_fraction, config.test_fraction
    )

    train_loader = _to_loader(X_tr, y_tr, config.batch_size, True)
    val_loader = _to_loader(X_val, y_val, config.batch_size, False)
    test_loader = _to_loader(X_te, y_te, config.batch_size, False)

    n_features = dataset.X.shape[-1]
    pred_horizon = dataset.y.shape[-1]

    device = torch.device(config.device)
    model = ForexLNN(n_features, units=config.units, pred_horizon=pred_horizon, dropout=config.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    mse_fn = torch.nn.MSELoss()
    bce_fn = torch.nn.BCEWithLogitsLoss()

    def combined_loss(returns_pred, dir_logit, y_target):
        mse = mse_fn(returns_pred, y_target)
        # Compute target direction from cumulative target return (in z-scored space, sign is preserved)
        cum = y_target.sum(dim=1)
        dir_target = (cum > 0).float()
        bce = bce_fn(dir_logit, dir_target)
        return mse + config.direction_loss_weight * bce, mse.item(), bce.item()

    best_val = float("inf")
    patience_left = config.early_stopping_patience
    history: list[dict] = []
    epochs_run = 0
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        epochs_run = epoch
        t0 = time.time()
        model.train()
        tr_total = []
        tr_mse = []
        tr_bce = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            r_pred, d_logit = model(xb)
            loss, mse_v, bce_v = combined_loss(r_pred, d_logit, yb)
            loss.backward()
            opt.step()
            tr_total.append(loss.item())
            tr_mse.append(mse_v)
            tr_bce.append(bce_v)

        model.eval()
        with torch.no_grad():
            vl = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                r_pred, d_logit = model(xb)
                loss, _, _ = combined_loss(r_pred, d_logit, yb)
                vl.append(loss.item())
        val_loss = float(np.mean(vl))
        train_loss = float(np.mean(tr_total))
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_mse": float(np.mean(tr_mse)), "train_bce": float(np.mean(tr_bce)),
            "secs": elapsed,
        })
        log.info(
            "epoch %d train=%.4f val=%.4f mse=%.4f bce=%.4f t=%.1fs",
            epoch, train_loss, val_loss, np.mean(tr_mse), np.mean(tr_bce), elapsed,
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_left = config.early_stopping_patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_features": n_features,
                    "units": config.units,
                    "pred_horizon": pred_horizon,
                    "dropout": config.dropout,
                    "feature_columns": FEATURE_COLUMNS,
                    "feature_means": dataset.feature_means.tolist(),
                    "feature_stds": dataset.feature_stds.tolist(),
                    "target_mean": dataset.target_mean,
                    "target_std": dataset.target_std,
                },
                save_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("early stopping at epoch %d", epoch)
                break

    # Test eval with best checkpoint
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    best_model = ForexLNN(n_features, units=config.units, pred_horizon=pred_horizon, dropout=0.0).to(device)
    best_model.load_state_dict(ckpt["model_state"])
    best_model.eval()

    pred_returns_all = []
    pred_probs_all = []
    targets_all = []
    test_losses = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb_d, yb_d = xb.to(device), yb.to(device)
            r_pred, d_logit = best_model(xb_d)
            loss, _, _ = combined_loss(r_pred, d_logit, yb_d)
            test_losses.append(loss.item())
            pred_returns_all.append(r_pred.cpu().numpy())
            pred_probs_all.append(torch.sigmoid(d_logit).cpu().numpy())
            targets_all.append(yb.numpy())

    if pred_returns_all:
        preds_ret = np.concatenate(pred_returns_all, axis=0)
        preds_prob = np.concatenate(pred_probs_all, axis=0)
        targs = np.concatenate(targets_all, axis=0)
        test_loss = float(np.mean(test_losses))

        dir_acc = _directional_accuracy(preds_ret, targs)

        # Classifier accuracy on all test samples
        targ_dir_bin = (targs.sum(axis=1) > 0).astype(int)
        pred_dir_bin = (preds_prob >= 0.5).astype(int)
        cls_acc = float((pred_dir_bin == targ_dir_bin).mean())

        # Confident-only accuracy (|prob - 0.5| >= 0.2 → confidence 70%+ in one class)
        confident_mask = np.abs(preds_prob - 0.5) >= 0.2
        if confident_mask.sum() > 0:
            cls_p70_acc = float((pred_dir_bin[confident_mask] == targ_dir_bin[confident_mask]).mean())
            cls_p70_n = int(confident_mask.sum())
        else:
            cls_p70_acc = 0.0
            cls_p70_n = 0
    else:
        test_loss = float("nan")
        dir_acc = 0.0
        cls_acc = 0.0
        cls_p70_acc = 0.0
        cls_p70_n = 0

    result = TrainResult(
        best_val_loss=best_val,
        test_loss=test_loss,
        test_directional_accuracy=dir_acc,
        test_classifier_accuracy=cls_acc,
        test_classifier_p70_acc=cls_p70_acc,
        test_classifier_p70_count=cls_p70_n,
        epochs_run=epochs_run,
        history=history,
        model_path=str(save_path),
    )

    with open(save_path.with_suffix(".json"), "w") as f:
        json.dump(asdict(result), f, indent=2)
    return result
