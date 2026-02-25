"""PyTorch LSTM for sequence-to-one forecasting."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from electricity_forecast.config import get_config
from electricity_forecast.models.base import ForecastModel

EXCLUDE_COLS = {"target", "lmp", "datetime", "datetime_begin", "timestamp"}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype in ("float64", "int64")]


class SeqDataset(Dataset):
    """Sequence dataset for LSTM: each sample is (seq, target)."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        feat_cols: list[str],
        target_col: str = "target",
    ) -> None:
        self.X = df[feat_cols].fillna(0).values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[i : i + self.seq_len])
        y = torch.tensor(self.y[i + self.seq_len - 1], dtype=torch.float32)
        return x, y


class LSTMModel(nn.Module):
    """LSTM sequence-to-one."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


class LSTMForecast(ForecastModel):
    """LSTM forecaster with early stopping."""

    def __init__(
        self,
        sequence_length: int = 168,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        **kwargs: Any,
    ) -> None:
        cfg = get_config()
        lstm_cfg = cfg.get("model", {}).get("lstm", {})
        self.seq_len = sequence_length or lstm_cfg.get("sequence_length", 168)
        self.hidden_size = hidden_size or lstm_cfg.get("hidden_size", 64)
        self.num_layers = num_layers or lstm_cfg.get("num_layers", 2)
        self.dropout = dropout or lstm_cfg.get("dropout", 0.2)
        self.epochs = epochs or lstm_cfg.get("epochs", 100)
        self.batch_size = batch_size or lstm_cfg.get("batch_size", 32)
        self.patience = early_stopping_patience or lstm_cfg.get("early_stopping_patience", 10)
        self.feature_names_: list[str] = []
        self.model_: LSTMModel | None = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, **kwargs: Any
    ) -> "LSTMForecast":
        feats = _feature_cols(train_df)
        self.feature_names_ = feats
        n_feat = len(feats)

        ds = SeqDataset(train_df, self.seq_len, feats)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model_ = LSTMModel(n_feat, self.hidden_size, self.num_layers, self.dropout).to(
            self.device_
        )
        opt = torch.optim.Adam(self.model_.parameters())
        best_loss = float("inf")
        wait = 0

        for _ep in range(self.epochs):
            self.model_.train()
            loss_sum = 0.0
            for x, y in loader:
                x, y = x.to(self.device_), y.to(self.device_)
                opt.zero_grad()
                pred = self.model_(x)
                loss = nn.functional.mse_loss(pred, y)
                loss.backward()
                opt.step()
                loss_sum += loss.item()
            _ = loss_sum / len(loader)  # train_loss for potential logging

            if val_df is not None and len(val_df) >= self.seq_len:
                val_pred = self._predict_internal(val_df, feats)
                val_loss = float(
                    np.mean((val_pred - val_df["target"].values[self.seq_len - 1 :]) ** 2)
                )
                if val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                if wait >= self.patience:
                    break
        return self

    def _predict_internal(self, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
        X = df[feats].fillna(0).values.astype(np.float32)
        seqs = [X[i : i + self.seq_len] for i in range(len(X) - self.seq_len + 1)]
        if not seqs:
            return np.array([])
        batch = torch.from_numpy(np.stack(seqs)).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(batch).cpu().numpy()
        return pred

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        feats = [c for c in self.feature_names_ if c in df.columns]
        if len(df) < self.seq_len:
            return pd.Series([float("nan")] * len(df), index=df.index)
        pred = self._predict_internal(df, feats)
        pad = [float("nan")] * (self.seq_len - 1)
        full = np.concatenate([pad, pred])
        return pd.Series(full[: len(df)], index=df.index)

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(
            {
                "state_dict": self.model_.state_dict() if self.model_ else None,
                "feature_names": self.feature_names_,
                "seq_len": self.seq_len,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LSTMForecast":
        import joblib

        data = joblib.load(path)
        m = cls(
            sequence_length=data["seq_len"],
            hidden_size=data["hidden_size"],
            num_layers=data["num_layers"],
            dropout=data["dropout"],
        )
        m.feature_names_ = data["feature_names"]
        m.model_ = LSTMModel(
            len(m.feature_names_),
            m.hidden_size,
            m.num_layers,
            m.dropout,
        )
        m.model_.load_state_dict(data["state_dict"])
        m.model_.eval()
        return m
