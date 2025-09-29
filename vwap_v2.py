# %%
# -*- coding: utf-8 -*-
# Python 3.9+
# pip install requests==2.32.3 python-dotenv==1.0.1 pybit==5.7.0

import os, time, hmac, hashlib, urllib.parse, requests, math, random
from dataclasses import dataclass
from typing import Optional, Tuple
from dotenv import load_dotenv
from pybit.unified_trading import HTTP as BybitHTTP
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING

# ================== パラメータ（ここだけ触れば戦略全体を調整できます） ==================
ASTER_SYMBOL_EXEC   = "ASTERUSDT"
BYBIT_ALT_SYMBOL    = "ASTERUSDT"
BYBIT_BTC_SYMBOL    = "BTCUSDT"

COEFF_WINDOW        = 10
USE_MAD_SCALE       = True
RESID_K             = 1.0
RESID_PERSIST       = 1
VWAP_SHORT          = 7
VWAP_MEDIUM         = 14
USE_DIRECTION_FILTER= False

NOTIONAL_USD        = 300.0
TAKER_FEE           = 0.00035
STOP_LOSS_PCT       = 0.040

KLINE_INTERVAL_BYBIT= "1"
KLINE_INTERVAL_ASTER= "1m"
KLINE_LIMIT         = 120
MINUTE_DELAY_SEC    = 5
RETRY_WAIT_SEC      = 5
MAX_RETRIES         = 12

ASTER_HOST     = "https://fapi.asterdex.com"
ASTER_KEY_HDR  = "X-MBX-APIKEY"
ASTER_PRICE_TICK   = Decimal("0.0001")
ASTER_PRICE_PREC   = 4
ASTER_QTY_STEP     = Decimal("0.01")
ASTER_QTY_MIN      = Decimal("0.01")
ASTER_QTY_PREC     = 2

USER_AGENT     = "AsterVwapResidualBot/1.0"
RECV_WINDOW    = 50000

# === 追加: SL後の挙動ポリシー ===
AFTER_SL_POLICY = "WAIT_CROSS"   # "IMMEDIATE" or "WAIT_CROSS"
COOLDOWN_SEC_AFTER_SL = 30       # SL直後にクールダウンする秒数

# ================== ユーティリティ ==================
def _now_ms() -> int: return int(time.time() * 1000)
def _hmac_sha256(secret: str, msg: str) -> str: return hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()
def _round_to_step(val: Decimal, step: Decimal, direction: str) -> Decimal:
    units = (val / step).to_integral_value(rounding=(ROUND_CEILING if direction == "up" else ROUND_FLOOR))
    return units * step
def _limit_decimals(val: Decimal, decimals: int) -> Decimal:
    if decimals <= 0: return Decimal(int(val))
    q = Decimal(1).scaleb(-decimals)
    return val.quantize(q, rounding=ROUND_FLOOR)
def round_down_qty(qty: float, step: float) -> float:
    q = Decimal(str(qty)); s = Decimal(str(step))
    units = (q / s).to_integral_value(rounding=ROUND_FLOOR)
    return float(units * s)
def last_closed_open_ms(now_ms: int) -> int:
    minute_ms = 60_000
    return ((now_ms - minute_ms) // minute_ms) * minute_ms
def pick_bar(rows: list, target_open_ms: int) -> Optional[dict]:
    for x in rows:
        if x["openTime"] == target_open_ms: return x
    return None
def sleep_until_next_minute_plus(delta_sec: int = MINUTE_DELAY_SEC):
    now = time.time()
    next_min = math.floor(now / 60.0) * 60.0 + 60.0 + delta_sec
    time.sleep(max(0.0, next_min - now))

# ================== ASTER v2 HMAC クライアント（発注先） ==================
@dataclass
class AsterClient:
    api_key: str
    api_secret: str
    host: str = ASTER_HOST
    price_tick: Decimal = ASTER_PRICE_TICK
    price_precision: int = ASTER_PRICE_PREC
    qty_step: Decimal = ASTER_QTY_STEP
    qty_min: Decimal = ASTER_QTY_MIN
    qty_precision: int = ASTER_QTY_PREC

    def _request(self, method: str, path: str, params: dict):
        url = self.host + path
        q = dict(params or {})
        q.setdefault("recvWindow", RECV_WINDOW)
        q["timestamp"] = _now_ms()
        qs = urllib.parse.urlencode(q, doseq=True)
        sig = _hmac_sha256(self.api_secret, qs)
        headers = {ASTER_KEY_HDR: self.api_key, "User-Agent": USER_AGENT}

        if method.upper() == "GET":
            full_url = f"{url}?{qs}&signature={sig}"
            r = requests.get(full_url, headers=headers, timeout=30)
        elif method.upper() == "POST":
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            body = f"{qs}&signature={sig}"
            r = requests.post(url, data=body, headers=headers, timeout=30)
        elif method.upper() == "DELETE":
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            body = f"{qs}&signature={sig}"
            r = requests.delete(url, data=body, headers=headers, timeout=30)
        else:
            raise ValueError("Unsupported method")

        if r.status_code >= 400:
            raise RuntimeError(f"Aster {method} {path} failed: {r.status_code} {r.text[:500]}")
        try:
            return r.json()
        except Exception:
            return r.text

    # precision helpers
    def fmt_qty(self, qty: float, direction: str = "down") -> str:
        v = Decimal(str(qty))
        v = _round_to_step(v, self.qty_step, direction)
        if v < max(self.qty_min, self.qty_step):
            v = self.qty_step
        v = _limit_decimals(v, self.qty_precision)
        return f"{v:.{self.qty_precision}f}".rstrip("0").rstrip(".")

    def fmt_price(self, price: float, direction: str) -> str:
        v = Decimal(str(price))
        v = _round_to_step(v, self.price_tick, direction)
        v = _limit_decimals(v, self.price_precision)
        return f"{v:.{self.price_precision}f}".rstrip("0").rstrip(".")

    # public
    def get_klines(self, symbol: str, interval: str = KLINE_INTERVAL_ASTER, limit: int = KLINE_LIMIT):
        d = self._request("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        rows = [{"openTime": int(r[0]), "open": float(r[1]), "high": float(r[2]),
                 "low": float(r[3]), "close": float(r[4])} for r in d]
        rows.sort(key=lambda x: x["openTime"])
        return rows

    def get_position(self, symbol: str):
        d = self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol})
        item = d[0] if isinstance(d, list) else d
        qty = float(item.get("positionAmt", 0.0))
        ep  = float(item.get("entryPrice", 0.0))
        return qty, ep

    def place_market(self, symbol: str, side: str, qty: float, reduce_only: bool = False):
        q_str = self.fmt_qty(qty, direction="down")
        p = {
            "symbol": symbol, "side": side, "type": "MARKET",
            "quantity": q_str, "positionSide": "BOTH"
        }
        if reduce_only:
            p["reduceOnly"] = "true"
        return self._request("POST", "/fapi/v1/order", p)

    def place_stop_market_close(self, symbol: str, trigger_price: float, side_to_close: str, fallback_qty: float = None):
        direction = "down" if side_to_close == "SELL" else "up"
        stop_str  = self.fmt_price(trigger_price, direction)
        pA = {"symbol": symbol, "side": side_to_close, "type": "STOP_MARKET",
              "stopPrice": stop_str, "closePosition": "true", "positionSide": "BOTH"}
        try:
            return self._request("POST", "/fapi/v1/order", pA)
        except Exception as e:
            # fallback: qty + reduceOnly
            if not fallback_qty or fallback_qty <= 0:
                pos = self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol})
                it = pos[0] if isinstance(pos, list) else pos
                fallback_qty = abs(float(it.get("positionAmt", 0.0))) or 0.0
            if fallback_qty <= 0:
                raise e
            q_str = self.fmt_qty(fallback_qty, direction="down")
            pB = {"symbol": symbol, "side": side_to_close, "type": "STOP_MARKET",
                  "stopPrice": stop_str, "quantity": q_str, "reduceOnly": "true", "positionSide": "BOTH"}
            return self._request("POST", "/fapi/v1/order", pB)

    def cancel_all(self, symbol: str):
        try:
            orders = self._request("GET", "/fapi/v1/openOrders", {"symbol": symbol})
            if isinstance(orders, list):
                for o in orders:
                    oid = o.get("orderId")
                    if oid is not None:
                        self._request("DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": oid})
        except Exception as e:
            print("[Aster] cancel_all error:", e)

# ================== Bybit クライアント（データ専用） ==================
@dataclass
class BybitClient:
    api_key: str
    api_secret: str
    testnet: bool = False
    def __post_init__(self):
        self.session = BybitHTTP(testnet=self.testnet, api_key=self.api_key, api_secret=self.api_secret)

    def get_klines(self, symbol: str, interval: str = KLINE_INTERVAL_BYBIT, limit: int = KLINE_LIMIT):
        r = self.session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
        lst = r["result"]["list"]
        rows = [{"openTime": int(rr[0]), "open": float(rr[1]), "high": float(rr[2]), "low": float(rr[3]), "close": float(rr[4])} for rr in lst]
        rows.sort(key=lambda x: x["openTime"])
        return rows

# ================== シグナル（VWAPクロス × 残差） ==================
def _rolling_resid_z(ret_alt: pd.Series, ret_btc: pd.Series, W: int, use_mad: bool) -> pd.Series:
    mA = ret_alt.rolling(W, min_periods=W).mean()
    mB = ret_btc.rolling(W, min_periods=W).mean()
    vB = ret_btc.rolling(W, min_periods=W).var(ddof=0)
    cAB = ret_alt.rolling(W, min_periods=W).cov(ret_btc)
    beta = cAB / vB
    alpha = mA - beta * mB
    pred = alpha + beta * ret_btc
    resid = ret_alt - pred
    if use_mad:
        def _mad(s):
            med = s.median()
            return 1.4826 * (np.abs(s - med)).median()
        scale = resid.rolling(W, min_periods=W).apply(_mad, raw=False)
    else:
        scale = resid.rolling(W, min_periods=W).std()
    return resid / scale

def compute_trade_signal(byb_alt_rows: list, byb_btc_rows: list) -> Tuple[Optional[int], dict]:
    """
    戻り値: (signal, debug)
      signal ∈ {1, 0, None}  … 1=ロング発火, 0=ショート発火, None=なし
      debug: 画面出力用の辞書（残差z, VWAP等）
    """
    # DataFrame化
    alt = pd.DataFrame(byb_alt_rows)
    btc = pd.DataFrame(byb_btc_rows)
    df = alt.merge(btc[["openTime","close"]].rename(columns={"close":"close_BTC"}), on="openTime", how="inner")
    if len(df) < max(COEFF_WINDOW, VWAP_MEDIUM) + 2:
        return None, {"reason": "insufficient bars"}

    # ログリターン
    df["ret_ALT"] = np.log(df["close"]).diff()
    df["ret_BTC"] = np.log(df["close_BTC"]).diff()

    # 残差 z
    df["resid_z"] = _rolling_resid_z(df["ret_ALT"], df["ret_BTC"], COEFF_WINDOW, USE_MAD_SCALE)

    # VWAP（ALT）
    tp = (alt["open"] + alt["high"] + alt["low"] + alt["close"]) / 4.0
    v   = 1.0  # Bybitの出来高が必要なら取得して組み込む。ここでは簡略化でTWAP近似→v=1
    df["VWAP_short"]  = tp.rolling(VWAP_SHORT,  min_periods=1).mean()
    df["VWAP_medium"] = tp.rolling(VWAP_MEDIUM, min_periods=1).mean()

    df["Regeme_vwap"] = (df["VWAP_short"] > df["VWAP_medium"]).astype(int)
    df["Reg_prev"]    = df["Regeme_vwap"].shift(1)
    df["ignition"]    = (df["Regeme_vwap"] != df["Reg_prev"])  # 変化バーが True

    # 残差フィルター
    gate_abs = df["resid_z"].abs() > RESID_K
    gate_persist = gate_abs.rolling(RESID_PERSIST, min_periods=RESID_PERSIST).sum() == RESID_PERSIST

    if USE_DIRECTION_FILTER:
        gate_dir = ((df["resid_z"] > 0) & (df["Regeme_vwap"] == 1)) | \
                   ((df["resid_z"] < 0) & (df["Regeme_vwap"] == 0))
        gate = gate_persist & gate_dir
    else:
        gate = gate_persist

    df["signal"] = np.where(df["ignition"] & gate, df["Regeme_vwap"], np.nan)

    # 直近確定足（最後の行）のシグナルだけを見る
    last = df.iloc[-1]
    sig  = int(last["signal"]) if pd.notna(last["signal"]) else None

    debug = {
        "resid_z": float(last["resid_z"]) if pd.notna(last["resid_z"]) else None,
        "vwap_s":  float(last["VWAP_short"]),
        "vwap_m":  float(last["VWAP_medium"]),
        "reg":     int(last["Regeme_vwap"]),
    }
    return sig, debug

# ================== 単一レッグ実行（ASTERのみ発注） ==================
@dataclass
class SingleLegState:
    side: Optional[str] = None
    qty: float = 0.0
    entry: Optional[float] = None
    sl: Optional[float] = None

class AsterSignalTrader:
    def __init__(self, aster: AsterClient, bybit: BybitClient):
        self.aster = aster
        self.bybit = bybit
        self.state = SingleLegState()
        self._cooldown_until = None
        self._waiting_for_cross = False
        self._sl_side = None


    def _fetch_last_closed_bars(self):
        """Bybit(ALT/BTC) と ASTER(ALT) の直近確定足 close を取る"""
        tries = 0
        while tries < MAX_RETRIES:
            now_ms = _now_ms()
            target = last_closed_open_ms(now_ms)
            try:
                alt_byb = self.bybit.get_klines(BYBIT_ALT_SYMBOL, KLINE_INTERVAL_BYBIT, KLINE_LIMIT)
                btc_byb = self.bybit.get_klines(BYBIT_BTC_SYMBOL, KLINE_INTERVAL_BYBIT, KLINE_LIMIT)
                alt_ast = self.aster.get_klines(ASTER_SYMBOL_EXEC, KLINE_INTERVAL_ASTER, KLINE_LIMIT)
            except Exception as e:
                print("[kline fetch error]", e)
                alt_byb, btc_byb, alt_ast = [], [], []

            ba = pick_bar(alt_byb, target)
            bb = pick_bar(btc_byb, target)
            ca = pick_bar(alt_ast, target)

            if ba and bb and ca:
                ts_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(target/1000))
                return alt_byb, btc_byb, alt_ast, ts_iso

            missing = []
            if not ba: missing.append("Bybit ALT closed")
            if not bb: missing.append("Bybit BTC closed")
            if not ca: missing.append("ASTER ALT closed")
            print(f"[retry in {RETRY_WAIT_SEC}s] missing={','.join(missing)} target_open:{target}")
            time.sleep(RETRY_WAIT_SEC); tries += 1

        return None, None, None, None

    def _compute_trade_qty(self, px: float) -> float:
        raw = NOTIONAL_USD / max(px, 1e-9)
        return max(round_down_qty(raw, float(ASTER_QTY_STEP)), float(ASTER_QTY_STEP))

    def _pnl_unrealized(self, cur_px: float) -> float:
        if not self.state.side or not self.state.entry or self.state.qty <= 0:
            return 0.0
        if self.state.side == "LONG":
            return (cur_px - self.state.entry) * self.state.qty
        else:
            return (self.state.entry - cur_px) * self.state.qty

    def _place_stop(self, side_to_close: str, trigger_price: float):
        try:
            self.aster.place_stop_market_close(
                ASTER_SYMBOL_EXEC, trigger_price, side_to_close=side_to_close, fallback_qty=self.state.qty
            )
        except Exception as e:
            print("[warn] stop setup failed:", e)

    def _enter(self, side: str, price: float):
        qty = self._compute_trade_qty(price)
        if qty <= 0:
            print("[enter] qty<=0, skip")
            return False
        self.aster.place_market(ASTER_SYMBOL_EXEC, "BUY" if side=="LONG" else "SELL", qty, reduce_only=False)
        self.state.side  = side
        self.state.qty   = qty
        self.state.entry = price
        if side == "LONG":
            self.state.sl = price * (1 - STOP_LOSS_PCT)
            self._place_stop("SELL", self.state.sl)
        else:
            self.state.sl = price * (1 + STOP_LOSS_PCT)
            self._place_stop("BUY", self.state.sl)
        print(f"[ENTER] {side} {qty} @ {price:.6f} (SL {self.state.sl:.6f})")
        return True

    def _reverse(self, side: str, price: float):
        # 既存を reduce-only でクローズ → すぐ反転新規
        if self.state.qty > 0 and self.state.side:
            self.aster.place_market(ASTER_SYMBOL_EXEC, "SELL" if self.state.side=="LONG" else "BUY",
                                    self.state.qty, reduce_only=True)
            print(f"[EXIT] close {self.state.side} {self.state.qty} @ ~{price:.6f}")
            self.state = SingleLegState()
        return self._enter(side, price)

    def _ensure_flat(self, price: float):
        if self.state.qty > 0 and self.state.side:
            self.aster.place_market(ASTER_SYMBOL_EXEC, "SELL" if self.state.side=="LONG" else "BUY",
                                    self.state.qty, reduce_only=True)
            print(f"[EXIT] flat {self.state.side} {self.state.qty} @ ~{price:.6f}")
            self.state = SingleLegState()

    def _reconcile_with_exchange(self):
        qty, ep = self.aster.get_position(ASTER_SYMBOL_EXEC)
        if abs(qty) < 1e-9 and self.state.qty > 0:
            print("[SL DETECTED] Position was liquidated/closed externally.")
            prev_side = self.state.side
            self.state = SingleLegState()
            self.aster.cancel_all(ASTER_SYMBOL_EXEC)
            self._cooldown_until = time.time() + COOLDOWN_SEC_AFTER_SL
            if AFTER_SL_POLICY == "IMMEDIATE":
                self._sl_side = prev_side
            else:
                self._waiting_for_cross = True

    def run(self):
        while True:
            try:
                sleep_until_next_minute_plus(MINUTE_DELAY_SEC)

                alt_byb, btc_byb, alt_ast, ts = self._fetch_last_closed_bars()
                if alt_byb is None: continue

                self._reconcile_with_exchange()  # === 追加 ===

                sig, dbg = compute_trade_signal(alt_byb, btc_byb)
                last_ast = alt_ast[-1]["close"]
                side_txt = self.state.side or "-"
                upnl = self._pnl_unrealized(last_ast)

                # ログ出力
                print(f"[{ts}] side={side_txt} entry={self.state.entry or '-'} "
                      f"price={last_ast:.6f} uPnL={upnl:+.4f} USD "
                      f"(resid_z={dbg.get('resid_z'):.3f}, VWAP_s={dbg.get('vwap_s'):.4f}, VWAP_m={dbg.get('vwap_m'):.4f})")

                # === 追加: SL後の挙動制御 ===
                if self._cooldown_until and time.time() < self._cooldown_until:
                    print("[COOLDOWN] skipping signals")
                    continue
                if AFTER_SL_POLICY == "IMMEDIATE" and self._sl_side:
                    print("[SL POLICY] re-enter immediately")
                    self._enter(self._sl_side, last_ast)
                    self._sl_side = None
                    continue
                if AFTER_SL_POLICY == "WAIT_CROSS" and self._waiting_for_cross:
                    if dbg.get("reg") is not None and dbg.get("reg") != (1 if self.state.side=="LONG" else 0):
                        print("[SL POLICY] cross detected, re-entry enabled")
                        self._waiting_for_cross = False
                    else:
                        continue

                if sig is None: continue
                desired = "LONG" if sig == 1 else "SHORT"
                if self.state.side is None:
                    self._enter(desired, last_ast)
                elif self.state.side != desired:
                    self._reverse(desired, last_ast)
            except Exception as e:
                print("[run-loop error]", e)
                time.sleep(RETRY_WAIT_SEC)

# ================== 起動 ==================
def main():
    load_dotenv("keys.env")
    aster = AsterClient(api_key=os.getenv("ASTER_API_KEY") or "", api_secret=os.getenv("ASTER_API_SECRET") or "")
    bybit = BybitClient(api_key=os.getenv("BYBIT_API_KEY") or "", api_secret=os.getenv("BYBIT_API_SECRET") or "",
                        testnet=(os.getenv("BYBIT_TESTNET","false").lower()=="true"))

    print("==============================================")
    print("[START] Aster VWAP×Residual Trader")
    print(f" Exec={ASTER_SYMBOL_EXEC} (ASTER), SignalAlt={BYBIT_ALT_SYMBOL} (Bybit), SignalBTC={BYBIT_BTC_SYMBOL} (Bybit)")
    print(f" COEFF_WINDOW={COEFF_WINDOW}, MAD={USE_MAD_SCALE}, K={RESID_K}, PERSIST={RESID_PERSIST}")
    print(f" VWAP(short/medium)=({VWAP_SHORT},{VWAP_MEDIUM}), DirFilter={USE_DIRECTION_FILTER}")
    print(f" NOTIONAL_USD={NOTIONAL_USD}, SL={STOP_LOSS_PCT:.2%}, Fee(1way)={TAKER_FEE:.5f}")
    print(f" AFTER_SL_POLICY={AFTER_SL_POLICY}, COOLDOWN={COOLDOWN_SEC_AFTER_SL}s")
    print("==============================================")

    trader = AsterSignalTrader(aster, bybit)
    trader.run()

if __name__ == "__main__":
    main()


# %%



