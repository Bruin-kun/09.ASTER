import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import AverageTrueRange


def generate_labels_and_features(df, RSI_window = 14, Stoch_RSI_window = 14, ATR_window = 14, Stoch_RSI_smooth_k = 3, Stoch_RSI_smooth_d = 3, Vol_scale = [24, 48, 96],twap_forward_period = 24, twap_lookback_period = 16):
    eps = 1e-12
    
    #Features

    # 1) PERP OHLCV log returns
    df['O_log'] = np.log((df['O'] + eps) / (df['O'].shift(1) + eps))
    df['H_log'] = np.log((df['H'] + eps) / (df['H'].shift(1) + eps))
    df['L_log'] = np.log((df['L'] + eps) / (df['L'].shift(1) + eps))
    df['C_log'] = np.log((df['C'] + eps) / (df['C'].shift(1) + eps))
    df['V_log'] = np.log((df['V'] + eps) / (df['V'].shift(1) + eps))  # ← 二重定義を整理

    #Labels
    
    #3.Generate TWAP
    df['repPrice'] = (df['O'] + df['H'] + df['L'] + df['C']) / 4
    x = df['repPrice'].to_numpy()
    twap = []
    N = len(x)
    for i in range(N):
        if i + twap_forward_period +1 <= N:
            twap.append(np.mean(x[i + 1 : i +1 + twap_forward_period]))
        else:
            twap.append(np.nan)
    df['forward_TWAP'] = twap

    df['past_TWAP'] = df['repPrice'].rolling(
        window=twap_lookback_period, min_periods=twap_lookback_period
    ).mean()
    #############################################################
    #4.Create TWAP Return vs past TWAP
    df['TWAP_return'] = (df['forward_TWAP'] / df['past_TWAP'])-1

    #5.Create label - Doten Style
    df['label'] = np.where(df['TWAP_return'] > 0, 1, 0)
    #############################################################
    
    # #############################################################
    # #4.Create TWAP Return vs C
    # df['TWAP_return'] = (df['forward_TWAP'] / df['C_perp'])-1

    # #5.Create label - Doten Style
    # df['label'] = np.where(df['TWAP_return'] > 0, 1, 0)
    # #############################################################

    df = df.dropna()

    return df


def add_trade_weights(
    df: pd.DataFrame,
    *,
    fee: float = 0.0011,
    slip: float = 0.0002,
    rem_horizon: int | None = None,   # None のときは forward_period 相当を推定
    quantile: float = 0.95,
    gamma: float = 0.995,
    alpha: float = 0.7,
    wmin: float = 0.0,
    wmax: float = 3.0,
    sigma_scale: float = 0.5
) -> pd.DataFrame:

    # ★ 位置インデックスに揃える（ここが最重要）
    out = df.reset_index(drop=True).copy()
    n = len(out)
    if n == 0:
        return out

    eps  = 1e-12
    cost = float(fee + slip)

    # -----------------------------
    # 1) ベース重み
    # -----------------------------
    r_net = np.log((out['forward_TWAP'].values + eps) / (out['past_TWAP'].values + eps)) - cost

    finite_mask = np.isfinite(r_net)
    if finite_mask.any():
        med = np.median(r_net[finite_mask])
        mad = np.median(np.abs(r_net[finite_mask] - med)) + eps
    else:
        med, mad = 0.0, 1.0

    sigma = mad * max(sigma_scale, 1e-6)

    w_base = np.where(np.abs(r_net) <= cost, 0.0, np.log1p(np.abs(r_net) / sigma))
    w_base[~np.isfinite(w_base)] = 0.0

    # -----------------------------
    # 2) 残余エッジ重み（位置ベースで計算）
    # -----------------------------
    # ランID（0/1が変わるたびに +1）
    run_id = (out['label'] != out['label'].shift(1)).cumsum()

    # repPrice の累積和（位置ベース）
    rep = out['repPrice'].astype(float).values
    S = np.concatenate([[0.0], np.cumsum(rep)])  # 長さ n+1 → S[k] は rep[0:k] の和

    # 残余をみる最大先行本数
    if rem_horizon is None:
        # forward_TWAP の末尾 NaN 連長から推定（簡便法）
        ft = out['forward_TWAP'].values
        tail_nans = 0
        for x in ft[::-1]:
            if not np.isfinite(x):
                tail_nans += 1
            else:
                break
        rem_horizon = max(1, tail_nans)

    rem_h = int(rem_horizon)
    rem_h = max(1, rem_h)

    w_rem = np.zeros(n, dtype=float)

    for rid, g in out.groupby(run_id, sort=False):
        idx_pos = g.index.to_numpy()              # ★ 0..N-1 の“位置”
        if idx_pos.size == 0:
            continue

        lab = int(out.loc[idx_pos[0], 'label'])   # そのランの方向
        e   = int(idx_pos[-1])                    # ラン終端（位置）

        for t in idx_pos:
            # 先行範囲 [t+1 .. min(e, t+rem_h)]（位置ベース）
            u_max = min(e, t + rem_h)
            if u_max <= t:
                continue

            past_t = float(out.at[t, 'past_TWAP'])
            if not np.isfinite(past_t) or past_t <= 0:
                continue

            ks = np.arange(1, (u_max - t) + 1, dtype=int)  # 1..(u_max - t)
            # TWAP(t->u) = (S[u+1] - S[t+1]) / (u - t)
            # S は長さ n+1 なので、最大参照は u_max+1 ≤ n
            numer = S[(t + ks) + 1] - S[t + 1]
            denom = ks.astype(float)
            twap_tu = numer / np.maximum(denom, 1.0)

            if gamma < 1.0:
                disc = (gamma ** ks)
            else:
                disc = 1.0

            if lab == 1:
                rem_arr = np.log(np.maximum(twap_tu, eps) / np.maximum(past_t, eps)) * disc - cost
            else:
                rem_arr = np.log(np.maximum(past_t, eps) / np.maximum(twap_tu, eps)) * disc - cost

            rem_arr = rem_arr[np.isfinite(rem_arr)]
            if rem_arr.size == 0:
                continue

            q = float(np.clip(quantile, 0.0, 1.0))
            # 分位極値（ヒゲ耐性）
            rem_best = np.quantile(rem_arr, q) if rem_arr.size > 1 else rem_arr[0]
            rem_best = max(0.0, float(rem_best))

            # ベースの sigma を流用して対数圧縮
            w_rem[t] = np.log1p(rem_best / sigma) if rem_best > 0 else 0.0

    # -----------------------------
    # 3) 合成・クリップ
    # -----------------------------
    w = w_base * (1.0 + alpha * w_rem)
    w = np.clip(w, wmin, wmax)
    w[~np.isfinite(w)] = 0.0

    out['weight_base'] = w_base
    out['weight_rem']  = w_rem
    out['weight']      = w

    # forward_TWAP / past_TWAP / repPrice がNaNの行は落とす
    out = out.dropna(subset=['forward_TWAP', 'past_TWAP', 'repPrice']).reset_index(drop=True)
    return out

