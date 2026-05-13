"""Training pipeline for the LNN."""
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


@dataclass
class TrainResult:
    best_val_loss: float
    test_loss: float
    test_directional_accuracy: float
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
    loss_fn = torch.nn.MSELoss()

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
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        model.eval()
        with torch.no_grad():
            vl = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl.append(loss_fn(model(xb), yb).item())
        val_loss = float(np.mean(vl))

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "secs": elapsed})
        log.info("epoch %d train=%.6f val=%.6f time=%.1fs", epoch, train_loss, val_loss, elapsed)

        if val_loss < best_val - 1e-8:
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
    preds = []
    targs = []
    test_losses = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb_d, yb_d = xb.to(device), yb.to(device)
            p = best_model(xb_d)
            test_losses.append(loss_fn(p, yb_d).item())
            preds.append(p.cpu().numpy())
            targs.append(yb.numpy())
    if preds:
        preds_arr = np.concatenate(preds, axis=0)
        targs_arr = np.concatenate(targs, axis=0)
        test_loss = float(np.mean(test_losses))
        dir_acc = _directional_accuracy(preds_arr, targs_arr)
    else:
        test_loss = float("nan")
        dir_acc = float("nan")

    result = TrainResult(
        best_val_loss=best_val,
        test_loss=test_loss,
        test_directional_accuracy=dir_acc,
        epochs_run=epochs_run,
        history=history,
        model_path=str(save_path),
    )

    # Save sidecar json
    with open(save_path.with_suffix(".json"), "w") as f:
        json.dump(asdict(result), f, indent=2)
    return result
