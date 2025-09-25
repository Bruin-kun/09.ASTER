# %%
# -*- coding: utf-8 -*-
# Python 3.9.6
# pip install requests==2.32.3 python-dotenv==1.0.1 pybit==5.7.0

import os, time, hmac, hashlib, urllib.parse, requests, math, threading
from dataclasses import dataclass
from typing import Optional, Tuple
from dotenv import load_dotenv
from pybit.unified_trading import HTTP as BybitHTTP
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import random

# ================== 設定 ==================
SYMBOL         = "ASTERUSDT"
NOTIONAL_USD   = 100          # 片側の想定USDサイズ（両所で同サイズ）
BTC_QTY_STEP   = 1          # 取引所の最小数量刻み（BTCは 0.001 が無難）
POLL_SEC       = 120            # 2分ごとチェック
REENTER_DELAY  = 60             # 1分後に再エントリー
CYCLE_RESET_S  = 60 * 60        # 1.5時間で一旦リセット
SL_PCT         = 0.1           # ±3% ストップレンジ
RECV_WINDOW    = 50000
USER_AGENT     = "FundingArbBot/1.0"

# ==== Kline & Scheduler ====
KLINE_INTERVAL_ASTER   = "1m"
KLINE_INTERVAL_BYBIT   = "1"      # Bybitの1分
KLINE_LIMIT            = 10        # 余裕を持って5本
MINUTE_DELAY_SEC       = 5        # 毎分 +3秒でリフレッシュ
RETRY_WAIT_SEC         = 5        # 確定足が見つからない時に待つ秒数
MAX_RETRIES            = 12       # 5秒×12=60秒まで粘る
# ================== スプレッド戦略パラメータ ==================
# エントリー/エグジットのしきい値（Aster/Bybit - 1）
PREM_ENTER = 0.002   # +0.20% 以上で Aster SHORT / Bybit LONG
PREM_EXIT  = -0.0001  # -0.20% 以下で Aster LONG  / Bybit SHORT

# ドテン＆片足解消ガード
ALWAYS_REVERSE = True         # 逆側シグナルで必ずドテン
LIQUIDATION_GUARD = True      # 片足だけ消えてたら即全クローズ（再エントリーは次分から）

# === In-band policy（PREM_EXIT <= premium <= PREM_ENTER の時の動作）===
# フラット時の動作: "BIAS" で入る / "FLAT" で様子見
IN_BAND_FLAT_ACTION = "BIAS"   # or "FLAT"

# 保有中の動作: "HOLD"（維持） / "BIAS"（BIASへドテン） / "FLAT"（クローズ）
IN_BAND_HOLD_ACTION = "HOLD"   # or "BIAS" or "FLAT"



ASTER_HOST     = "https://fapi.asterdex.com"  # v2 HMAC
ASTER_KEY_HDR  = "X-MBX-APIKEY"

# Bybit endpoints are handled by pybit.

USE_MANUAL_PRECISION = True

ASTER_PRECISION = {
    "price_tick": Decimal("0.1"),
    "price_precision": 1,
    "qty_step": Decimal("0.001"),
    "qty_min": Decimal("0.001"),
    "qty_precision": 3,
}

BYBIT_PRECISION = {
    "price_tick": Decimal("0.1"),
    "price_precision": 2,
    "qty_step": Decimal("0.001"),
    "qty_min": Decimal("0.001"),
    "qty_precision": 3,
}


# ================== 共通ユーティリティ ==================
def _now_ms() -> int:
    return int(time.time() * 1000)

def _hmac_sha256(secret: str, msg: str) -> str:
    return hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()

def _round_to_step(val: Decimal, step: Decimal, direction: str) -> Decimal:
    units = (val / step).to_integral_value(rounding=(ROUND_CEILING if direction == "up" else ROUND_FLOOR))
    return units * step

def _limit_decimals(val: Decimal, decimals: int) -> Decimal:
    if decimals <= 0:
        return Decimal(int(val))
    q = Decimal(1).scaleb(-decimals)  # 10^-decimals
    # 超過桁は切り捨て（上方向が必要なら呼び出し側で step 丸めを up にする）
    return val.quantize(q, rounding=ROUND_FLOOR)

def _round_to_step_dec(val: Decimal, step: Decimal, direction: str) -> Decimal:
    units = (val / step).to_integral_value(rounding=(ROUND_CEILING if direction == "up" else ROUND_FLOOR))
    return units * step

def _limit_decimals_dec(val: Decimal, decimals: int) -> Decimal:
    if decimals <= 0: return Decimal(int(val))
    q = Decimal(1).scaleb(-decimals)
    return val.quantize(q, rounding=ROUND_FLOOR)

def round_down_qty(qty: float, step: float) -> float:
    q = Decimal(str(qty))
    s = Decimal(str(step))
    units = (q / s).to_integral_value(rounding=ROUND_FLOOR)
    return float(units * s)

def last_closed_open_ms(now_ms: int) -> int:
    """直近で確定している1分足の openTime(ms) を返す"""
    # 例: now=14:37:12 → last_closed_open=14:36:00.000
    minute_ms = 60_000
    return ((now_ms - minute_ms) // minute_ms) * minute_ms

def pick_bar(rows: list, target_open_ms: int) -> Optional[dict]:
    """rows から openTime==target_open_ms のバーを返す。なければ None"""
    for x in rows:
        if x["openTime"] == target_open_ms:
            return x
    return None

def sleep_until_next_minute_plus(delta_sec: int = MINUTE_DELAY_SEC):
    now = time.time()
    next_min = math.floor(now / 60.0) * 60.0 + 60.0 + delta_sec
    to_sleep = max(0.0, next_min - now)
    time.sleep(to_sleep)

def _retry(fn, *args, retries=3, wait=1.0, backoff=2.0, jitter=0.2, **kwargs):
    """
    短期的な 5xx/ネットワーク失敗向けの汎用リトライ。
    例: _retry(self.aster.place_market, SYMBOL, "SELL", qty, retries=3, wait=0.5)
    """
    last_err = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            # 503/5xxはよくあるので待って再試行
            slp = wait * (backoff ** i) * (1.0 + random.uniform(-jitter, jitter))
            print(f"[retry {i+1}/{retries}] {fn.__name__} failed: {e}. sleep {slp:.2f}s")
            time.sleep(max(0.1, slp))
    raise last_err


# ================== Aster v2 HMAC クライアント ==================
@dataclass
class AsterClient:
    api_key: str
    api_secret: str
    host: str = ASTER_HOST
    # manual precision（必須）
    price_tick: Decimal = Decimal("0.1")
    price_precision: int = 1
    qty_step: Decimal = Decimal("0.001")
    qty_min: Decimal = Decimal("0.001")
    qty_precision: int = 3

    # ---- HTTP ----
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

    # ---- precision helpers ----
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

    # ---- public APIs ----
    def get_funding_rate(self, symbol: str) -> float:
        d = self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        item = d[0] if isinstance(d, list) else d
        return float(item.get("lastFundingRate") or item.get("fundingRate") or 0.0)

    def get_mark_price(self, symbol: str) -> float:
        d = self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        item = d[0] if isinstance(d, list) else d
        return float(item["markPrice"])

    def get_position(self, symbol: str):
        d = self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol})
        item = d[0] if isinstance(d, list) else d
        qty = float(item.get("positionAmt", 0.0))   # +long / -short
        ep  = float(item.get("entryPrice", 0.0))
        return qty, ep

    def place_market(self, symbol: str, side: str, qty: float, reduce_only: bool = False):
        q_str = self.fmt_qty(qty, direction="down")
        p = {
            "symbol": symbol,
            "side": side,  # "BUY"/"SELL"
            "type": "MARKET",
            "quantity": q_str,
            "positionSide": "BOTH",
        }
        if reduce_only:
            p["reduceOnly"] = "true"
        return self._request("POST", "/fapi/v1/order", p)

    def place_stop_market_close(self, symbol: str, trigger_price: float, side_to_close: str, fallback_qty: float = None):
        # SELL=ロング解消→下方向丸め, BUY=ショート解消→上方向丸め
        direction = "down" if side_to_close == "SELL" else "up"
        stop_str  = self.fmt_price(trigger_price, direction=direction)

        # A: closePosition=true（reduceOnlyは一緒に送らない）
        pA = {
            "symbol": symbol, "side": side_to_close, "type": "STOP_MARKET",
            "stopPrice": stop_str, "closePosition": "true", "positionSide": "BOTH",
        }
        try:
            return self._request("POST", "/fapi/v1/order", pA)
        except RuntimeError as e:
            print("[Aster STOP closePosition] failed, fallback to qty+reduceOnly:", e)

        # B: quantity + reduceOnly=true
        if not fallback_qty or fallback_qty <= 0:
            try:
                pos = self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol})
                item = pos[0] if isinstance(pos, list) else pos
                fallback_qty = abs(float(item.get("positionAmt", 0.0)))
            except Exception:
                fallback_qty = 0.0
        if fallback_qty <= 0:
            raise RuntimeError("Fallback stop failed and no open position qty found.")

        q_str = self.fmt_qty(fallback_qty, direction="down")
        pB = {
            "symbol": symbol, "side": side_to_close, "type": "STOP_MARKET",
            "stopPrice": stop_str, "quantity": q_str, "reduceOnly": "true", "positionSide": "BOTH",
        }
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

    def get_klines(self, symbol: str, interval: str = KLINE_INTERVAL_ASTER, limit: int = KLINE_LIMIT):
        """
        返却は 昇順（古→新）。
        Asterは /fapi/v1/klines が古→新で返る想定だが、念のため昇順に整列。
        """
        d = self._request("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        rows = []
        for r in d:
            # r = [openTime, open, high, low, close, ...]
            rows.append({
                "openTime": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low":  float(r[3]),
                "close":float(r[4]),
            })
        rows.sort(key=lambda x: x["openTime"])
        return rows

# ================== Bybit クライアント（pybit） ==================
@dataclass
class BybitClient:
    api_key: str
    api_secret: str
    testnet: bool = False
    # manual precision（必須）
    price_tick: Decimal = Decimal("0.5")
    price_precision: int = 1
    qty_step: Decimal = Decimal("0.001")
    qty_min: Decimal = Decimal("0.001")
    qty_precision: int = 3

    def __post_init__(self):
        self.session = BybitHTTP(testnet=self.testnet, api_key=self.api_key, api_secret=self.api_secret)

    # precision helpers
    def fmt_qty(self, qty: float, direction: str = "down") -> str:
        v = Decimal(str(qty))
        v = _round_to_step_dec(v, self.qty_step, direction)
        if v < max(self.qty_min, self.qty_step):
            v = self.qty_step
        v = _limit_decimals_dec(v, self.qty_precision)
        return f"{v:.{self.qty_precision}f}".rstrip("0").rstrip(".")

    def fmt_price(self, price: float, direction: str) -> str:
        v = Decimal(str(price))
        v = _round_to_step_dec(v, self.price_tick, direction)
        v = _limit_decimals_dec(v, self.price_precision)
        return f"{v:.{self.price_precision}f}".rstrip("0").rstrip(".")

    # public APIs
    def get_funding_rate(self, symbol: str) -> float:
        r = self.session.get_tickers(category="linear", symbol=symbol)
        return float(r["result"]["list"][0].get("fundingRate", 0.0))

    def get_mark_price(self, symbol: str) -> float:
        r = self.session.get_tickers(category="linear", symbol=symbol)
        return float(r["result"]["list"][0]["markPrice"])

    def get_position(self, symbol: str):
        r = self.session.get_positions(category="linear", symbol=symbol)
        lst = r["result"]["list"]
        if not lst:
            return 0.0, 0.0
        pos = lst[0]
        size = float(pos.get("size", 0.0))
        side = pos.get("side", "Buy")
        qty = size if side == "Buy" else -size
        ep  = float(pos.get("avgPrice", 0.0))
        return qty, ep

    def place_market(self, symbol: str, side: str, qty: float, reduce_only: bool = False):
        q_str = self.fmt_qty(qty)
        return self.session.place_order(
            category="linear",
            symbol=symbol,
            side=side.title(),       # "Buy"/"Sell"
            orderType="Market",
            qty=q_str,
            reduceOnly=reduce_only
        )

    def place_stop_market_close(self, symbol: str, trigger_price: float, side_to_close: str, qty: float):
        # SELL=long close -> 下方向, BUY=short close -> 上方向
        direction = "down" if side_to_close == "SELL" else "up"
        trig = self.fmt_price(trigger_price, direction)
        q_str = self.fmt_qty(qty)               # ★ StopOrder にも数量を明示
        trigger_dir = 2 if side_to_close == "SELL" else 1

        # 注意: qty を送る場合は closePosition は送らない（競合するため）
        return self.session.place_order(
            category="linear",
            symbol=symbol,
            side=side_to_close.title(),         # "Buy"/"Sell"
            orderType="Market",
            qty=q_str,                           # ★必須
            reduceOnly=True,
            closeOnTrigger=True,
            tpslMode="Full",
            triggerPrice=trig,
            triggerDirection=trigger_dir,
            timeInForce="GoodTillCancel",
            orderFilter="StopOrder",
            triggerBy="MarkPrice",
            positionIdx=0,                       # 片面/BOTH
        )
    
    def cancel_all(self, symbol: str):
        try:
            self.session.cancel_all_orders(category="linear", symbol=symbol)
        except Exception as e:
            print("[Bybit] cancel_all error:", e)

    def get_klines(self, symbol: str, interval: str = KLINE_INTERVAL_BYBIT, limit: int = KLINE_LIMIT):
        """
        返却は 昇順（古→新）。
        Bybitは新→古で返ることがあるので、反転して昇順にする。
        """
        r = self.session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
        lst = r["result"]["list"]
        rows = []
        for rr in lst:
            # rr = [start, open, high, low, close, ...] いずれも文字列のことが多い
            rows.append({
                "openTime": int(rr[0]),
                "open": float(rr[1]),
                "high": float(rr[2]),
                "low":  float(rr[3]),
                "close":float(rr[4]),
            })
        rows.sort(key=lambda x: x["openTime"])
        return rows

# ================== 戦略ロジック ==================
@dataclass
class Leg:
    ex_name: str                  # "ASTER" / "BYBIT"
    side: str                     # "LONG"/"SHORT"
    qty: float
    entry_price: float
    sl_price: float
    opened_ms: int

class FundingArbBot:
    def __init__(self, aster: AsterClient, bybit: BybitClient):
        self.aster = aster
        self.bybit = bybit
        self.legs: Tuple[Optional[Leg], Optional[Leg]] = (None, None)
        self.regime: Optional[str] = None   # "MAIN" / "BIAS" / None

    def _decide_sides_by_premium(self, premium: float) -> tuple[str, str, str]:
        """
        premium >  PREM_ENTER → MAIN:  Aster SHORT / Bybit LONG
        premium <  PREM_EXIT  → MAIN:  Aster LONG  / Bybit SHORT
        PREM_EXIT <= premium <= PREM_ENTER → BIAS: Aster LONG / Bybit SHORT
        戻り値: (side_a, side_b, regime)  regime ∈ {"MAIN","BIAS"}
        """
        if premium > PREM_ENTER:
            return ("SHORT", "LONG", "MAIN")
        elif premium < PREM_EXIT:
            return ("LONG", "SHORT", "MAIN")
        else:
            return ("LONG", "SHORT", "BIAS")
    
    def _decide_sides(self, fr_aster: float, fr_bybit: float) -> Tuple[str, str]:
        # 高い方をショート、低い方をロング
        if fr_aster > fr_bybit:
            return ("SHORT", "LONG")   # Aster, Bybit
        elif fr_aster < fr_bybit:
            return ("LONG", "SHORT")
        else:
            # 同値なら何もしない or どちらか固定。ここでは固定で Aster=SHORT, Bybit=LONG
            return ("SHORT", "LONG")

    def _compute_qty(self, price: float) -> float:
        raw = NOTIONAL_USD / price
        return max(round_down_qty(raw, BTC_QTY_STEP), BTC_QTY_STEP)

    def _place_leg(self, ex: str, side: str, qty: float, price_now: float) -> Leg:
        # どの取引所か
        if ex == "ASTER":
            cli = self.aster
        else:
            cli = self.bybit

        if side == "LONG":
            # 成行でロング
            cli.place_market(SYMBOL, "BUY", qty, reduce_only=False)
            sl_price = price_now * (1 - SL_PCT)
            # SLはロング解消=SELL
            try:
                if ex == "ASTER":
                    self.aster.place_stop_market_close(SYMBOL, sl_price, side_to_close="SELL", fallback_qty=qty)
                else:
                    self.bybit.place_stop_market_close(SYMBOL, sl_price, side_to_close="SELL", qty=qty)
            except Exception as e:
                print(f"[{ex} STOP(LONG close) setup failed]", e)
        else:
            # 成行でショート
            cli.place_market(SYMBOL, "SELL", qty, reduce_only=False)
            sl_price = price_now * (1 + SL_PCT)
            # SLはショート解消=BUY
            try:
                if ex == "ASTER":
                    self.aster.place_stop_market_close(SYMBOL, sl_price, side_to_close="BUY", fallback_qty=qty)
                else:
                    self.bybit.place_stop_market_close(SYMBOL, sl_price, side_to_close="BUY", qty=qty)
            except Exception as e:
                print(f"[{ex} STOP(SHORT close) setup failed]", e)

        # ★どの経路でも必ず返す
        return Leg(ex, side, qty, price_now, sl_price, _now_ms())


    def _get_positions(self) -> Tuple[float, float, float, float]:
        # qty(+long/-short), entryPrice
        a_qty, a_ep = self.aster.get_position(SYMBOL)
        b_qty, b_ep = self.bybit.get_position(SYMBOL)
        return a_qty, a_ep, b_qty, b_ep

    def _is_open(self, qty: float) -> bool:
        return abs(qty) > 1e-9

    def open_basket(self, premium: float, c_ast: float, c_byb: float, ts: str):
        # 望ましい方向を決定（MAIN/BIAS は既存ロジックのまま）
        if premium > PREM_ENTER:
            side_a, side_b, regime = "SHORT", "LONG", "MAIN"
        elif premium < PREM_EXIT:
            side_a, side_b, regime = "LONG", "SHORT", "MAIN"
        else:
            side_a, side_b, regime = "LONG", "SHORT", "BIAS"

        print(f"[price {ts}] Aster_close={c_ast:.6f}, Bybit_close={c_byb:.6f} → premium={premium:+.6%} ({regime})")

        qty_a = self._compute_qty(c_ast)
        qty_b = self._compute_qty(c_byb)

        # ★ 安全オープン：片脚だけ残さない
        ok = self._safe_open_pair(side_a, side_b, qty_a, qty_b, c_ast, c_byb, ts)
        if not ok:
            print(f"[enter-failed {ts}] pair entry aborted; will retry next minute")
            self.legs = (None, None)
            self.regime = None
            return

        # legs メタの更新（SL価格は概算）
        leg_a = Leg("ASTER", side_a, qty_a, c_ast, c_ast*(1-SL_PCT if side_a=="LONG" else 1+SL_PCT), _now_ms())
        leg_b = Leg("BYBIT", side_b, qty_b, c_byb, c_byb*(1-SL_PCT if side_b=="LONG" else 1+SL_PCT), _now_ms())
        self.legs = (leg_a, leg_b)
        self.regime = regime

        print(f"[enter {ts}] regime={regime} prem={premium:+.3%} | "
            f"Aster {side_a} {qty_a} @~{c_ast:.6f} | Bybit {side_b} {qty_b} @~{c_byb:.6f}")


    def _safe_open_pair(self, side_a: str, side_b: str, qty_a: float, qty_b: float,
                        px_a: float, px_b: float, ts: str) -> bool:
        """
        できるだけ「両脚同時に開く」ための安全実装。
        - 先に Bybit を建てる
        - 次に Aster を建てる（503などはリトライ）
        - Aster で失敗したら Bybit を reduce-only で巻き戻す
        """
        byb_ok = False
        try:
            # 1) Bybit leg
            _retry(self.bybit.place_market, SYMBOL,
                   "BUY" if side_b == "LONG" else "SELL", qty_b,
                   retries=3, wait=0.5)
            byb_ok = True

            # 2) Aster leg（失敗しやすいのでリトライ厚め）
            _retry(self.aster.place_market, SYMBOL,
                   "BUY" if side_a == "LONG" else "SELL", qty_a,
                   retries=5, wait=0.8)

            # 3) SL注文（失敗しても続行）
            try:
                if side_a == "LONG":
                    self.aster.place_stop_market_close(SYMBOL, px_a*(1-SL_PCT), side_to_close="SELL", fallback_qty=qty_a)
                else:
                    self.aster.place_stop_market_close(SYMBOL, px_a*(1+SL_PCT), side_to_close="BUY",  fallback_qty=qty_a)
            except Exception as e:
                print("[warn] Aster SL setup failed:", e)
            try:
                if side_b == "LONG":
                    self.bybit.place_stop_market_close(SYMBOL, px_b*(1-SL_PCT), side_to_close="SELL", qty=qty_b)
                else:
                    self.bybit.place_stop_market_close(SYMBOL, px_b*(1+SL_PCT), side_to_close="BUY",  qty=qty_b)
            except Exception as e:
                print("[warn] Bybit SL setup failed:", e)

            return True

        except Exception as e:
            print(f"[safe_open_pair ERR {ts}] {e}")
            # 失敗時：Bybit 側だけ建っていたら巻き戻す
            if byb_ok:
                try:
                    self.bybit.place_market(SYMBOL,
                        "SELL" if side_b == "LONG" else "BUY", qty_b, reduce_only=True)
                except Exception as e2:
                    print("[rollback Bybit] failed:", e2)
            return False

    def close_all(self, reason: str):
        print(f"[close_all] reason={reason}")
        # 1) まず両所のオープン注文をキャンセル（ストップ等が残っていても邪魔しない）
        try:
            self.aster.cancel_all(SYMBOL)
        except Exception as e:
            print("[Aster cancel_all] err:", e)
        try:
            self.bybit.cancel_all(SYMBOL)
        except Exception as e:
            print("[Bybit cancel_all] err:", e)

        # 2) ポジション reduce-only でクローズ（各3回リトライ）
        for ex in ("ASTER", "BYBIT"):
            try:
                a_qty, _, b_qty, _ = self._get_positions()
                if ex == "ASTER":
                    qty = a_qty
                    if self._is_open(qty):
                        side = "SELL" if qty > 0 else "BUY"
                        _retry(self.aster.place_market, SYMBOL, side, abs(qty), reduce_only=True,
                            retries=3, wait=0.5)
                else:
                    qty = b_qty
                    if self._is_open(qty):
                        side = "SELL" if qty > 0 else "BUY"
                        _retry(self.bybit.place_market, SYMBOL, side, abs(qty), reduce_only=True,
                            retries=3, wait=0.5)
            except Exception as e:
                print(f"[{ex} reduce-only close] err:", e)

        # 3) 仕上げ：もう一度オープン注文キャンセル
        try:
            self.aster.cancel_all(SYMBOL)
        except Exception as e:
            print("[Aster cancel_all 2] err:", e)
        try:
            self.bybit.cancel_all(SYMBOL)
        except Exception as e:
            print("[Bybit cancel_all 2] err:", e)

        # 4) ベリファイ：まだ片脚残っていたら警告（次ループで LIQUIDATION_GUARD が拾う）
        a_qty, _, b_qty, _ = self._get_positions()
        if self._is_open(a_qty) or self._is_open(b_qty):
            print(f"[warn] close_all done but remains: Aster={a_qty}, Bybit={b_qty}")

        self.legs = (None, None)
        self.regime = None



    def check_cycle_reset(self) -> bool:
        # 1.5h 経過で True
        l1, l2 = self.legs
        if not l1 or not l2:
            return False
        opened = min(l1.opened_ms, l2.opened_ms)
        return (_now_ms() - opened) >= CYCLE_RESET_S * 1000

    def poll_once(self) -> Optional[str]:
        # 片側が消えていたら "REENTER"、両方OKなら None
        a_qty, _, b_qty, _ = self._get_positions()
        a_open = self._is_open(a_qty)
        b_open = self._is_open(b_qty)

        if a_open and b_open:
            return None
        if (a_open and not b_open) or (b_open and not a_open):
            return "REENTER"
        # 両方消えてたら、そのまま再エントリー合図
        return "REENTER"

    def _fetch_last_closed_closes_with_retry(self) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        直近確定足の close を両所から取得。見つからなければ 5秒待って再トライ（最大 MAX_RETRIES）。
        戻り値: (close_ast, close_byb, iso_ts)  いずれか欠損なら None を返す
        """
        tries = 0
        while tries < MAX_RETRIES:
            now_ms = _now_ms()
            target = last_closed_open_ms(now_ms)

            try:
                kl_a = self.aster.get_klines(SYMBOL, KLINE_INTERVAL_ASTER, KLINE_LIMIT)
                kl_b = self.bybit.get_klines(SYMBOL, KLINE_INTERVAL_BYBIT, KLINE_LIMIT)
            except Exception as e:
                print("[kline fetch error]", e)
                kl_a, kl_b = [], []

            ba = pick_bar(kl_a, target)
            bb = pick_bar(kl_b, target)

            if ba and bb:
                ts_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(target/1000))
                return ba["close"], bb["close"], ts_iso

            # どちらか欠損 → 5秒待って再トライ
            missing = []
            if not ba: missing.append("Aster-closed-bar")
            if not bb: missing.append("Bybit-closed-bar")
            print(f"[retry in {RETRY_WAIT_SEC}s] missing={','.join(missing)} target_open:{target}")
            time.sleep(RETRY_WAIT_SEC)
            tries += 1

        return None, None, None

    def _calc_leg_upnl(self, entry_price: float, side: str, qty: float, last_close: float) -> Tuple[float, float]:
        """
        片脚の未実現PnL（USD, %）を計算。%は entry_price*qty に対する割合。
        """
        if qty is None or entry_price is None or side is None:
            return 0.0, 0.0
        if side.upper() == "LONG":
            pnl = (last_close - entry_price) * qty
        else:
            pnl = (entry_price - last_close) * qty
        denom = max(1e-12, entry_price * qty)
        return pnl, pnl / denom

    def hedge_break_detected(self) -> bool:
        """片側だけポジションが残っている（= ロスカ/SL/手動解消含む）かを検知"""
        a_qty, _, b_qty, _ = self._get_positions()
        a_open = self._is_open(a_qty)
        b_open = self._is_open(b_qty)
        return (a_open != b_open)  # XOR

    def run(self):
        while True:
            try:
                sleep_until_next_minute_plus(MINUTE_DELAY_SEC)

                c_ast, c_byb, ts = self._fetch_last_closed_closes_with_retry()
                if c_ast is None or c_byb is None:
                    print("[skip] closed bars not found after retries")
                    continue

                premium = (c_ast / c_byb) - 1.0
                print(f"[calc {ts}] Aster={c_ast:.6f}, Bybit={c_byb:.6f} → premium={premium:+.6%}")

                if LIQUIDATION_GUARD and self.hedge_break_detected():
                    print(f"[hedge_break {ts}] one side missing → closing all")
                    self.close_all("hedge_break")
                    continue

                in_band = (PREM_EXIT <= premium <= PREM_ENTER)
                legA, legB = self.legs
                has_pos = (legA is not None and legB is not None)

                if not has_pos:
                    # フラット → MAIN条件 or BIAS条件で入る
                    self.open_basket(premium, c_ast, c_byb, ts)
                else:
                    # regime に基づくドテン/HOLD（あなたの前回ロジックのままでOK）
                    # --- 略（既存の MAIN/BIAS 分岐） ---
                    ...
                if self.check_cycle_reset():
                    self.close_all("cycle_reset_1.5h")

            except Exception as loop_err:
                print("[run-loop error]", loop_err)
                # 念のため次分まで待ち、次の確定足で続行
                time.sleep(RETRY_WAIT_SEC)



# ================== 起動 ==================
def main():
    load_dotenv("keys.env")
    aster_kwargs = {}
    bybit_kwargs = {}

    if USE_MANUAL_PRECISION:
        aster_kwargs.update(
            price_tick=ASTER_PRECISION["price_tick"],
            price_precision=ASTER_PRECISION["price_precision"],
            qty_step=ASTER_PRECISION["qty_step"],
            qty_min=ASTER_PRECISION["qty_min"],
            qty_precision=ASTER_PRECISION["qty_precision"],
        )
        bybit_kwargs.update(
            price_tick=BYBIT_PRECISION["price_tick"],
            price_precision=BYBIT_PRECISION["price_precision"],
            qty_step=BYBIT_PRECISION["qty_step"],
            qty_min=BYBIT_PRECISION["qty_min"],
            qty_precision=BYBIT_PRECISION["qty_precision"],
        )

    aster = AsterClient(
        api_key=os.getenv("ASTER_API_KEY") or "",
        api_secret=os.getenv("ASTER_API_SECRET") or "",
        **aster_kwargs
    )
    bybit = BybitClient(
        api_key=os.getenv("BYBIT_API_KEY") or "",
        api_secret=os.getenv("BYBIT_API_SECRET") or "",
        testnet=(os.getenv("BYBIT_TESTNET", "false").lower() == "true"),
        **bybit_kwargs
    )
    bot = FundingArbBot(aster, bybit)

    # ★ 起動メッセージ
    print("==============================================")
    print("[START] PremiumArbBot booting...")
    print(f" Symbol={SYMBOL}")
    print(f" Kline(Aster)={KLINE_INTERVAL_ASTER}, Kline(Bybit)={KLINE_INTERVAL_BYBIT}, limit={KLINE_LIMIT}")
    print(f" PREM_ENTER={PREM_ENTER:+.4%}, PREM_EXIT={PREM_EXIT:+.4%}")
    print(f" NOTIONAL_USD={NOTIONAL_USD}, QTY_STEP={BTC_QTY_STEP}, SL_PCT={SL_PCT:.2%}")
    print(f" ALWAYS_REVERSE={ALWAYS_REVERSE}, LIQUIDATION_GUARD={LIQUIDATION_GUARD}")
    print("==============================================")

    bot.run()

if __name__ == "__main__":
    main()


# %%



