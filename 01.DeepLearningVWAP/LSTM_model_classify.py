import os, glob, datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from collections import deque

def make_latest_lstm_window(df_features, lookback, Features_list):
    """
    df_features: timestamp + feature列のDataFrame
    return: X shape (1, lookback, n_features), latest_window_start_ts, latest_window_end_ts
    """
    assert all(c in df_features.columns for c in Features_list), "特徴列が不足しています"

    if len(df_features) < lookback:
        raise ValueError(f"データ不足: len(df)={len(df_features)} < lookback={lookback}")

    # 末尾から lookback 行だけ
    tail = df_features.iloc[-lookback:].copy()
    ts_start = tail['timestamp'].iloc[0]
    ts_end   = tail['timestamp'].iloc[-1]

    X = tail[Features_list].to_numpy(dtype=np.float32)          # (lookback, n_features)
    X = np.expand_dims(X, axis=0)                              # (1, lookback, n_features)
    return X, ts_start, ts_end

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits
    
def load_classify_checkpoint(model_path, device="cpu",
                             input_dim=5, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.2):
    ckpt = torch.load(model_path, map_location=device)  # ← weights_only を外す
    params = ckpt.get("params", {})
    model = LSTMClassifier(
        input_dim=params.get("input_dim", input_dim),
        hidden_dim=params.get("hidden", hidden_dim),
        num_layers=params.get("layers", num_layers),
        num_classes=params.get("num_classes", num_classes),
        dropout=params.get("dropout", dropout)
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, params, ckpt

def load_scaler(scaler_path):
    return joblib.load(scaler_path)