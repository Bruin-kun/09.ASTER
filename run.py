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

# ================== 設定 ==================
SYMBOL         = "BTCUSDT"
NOTIONAL_USD   = 1000          # 片側の想定USDサイズ（両所で同サイズ）
BTC_QTY_STEP   = 0.001          # 取引所の最小数量刻み（BTCは 0.001 が無難）
POLL_SEC       = 120            # 2分ごとチェック
REENTER_DELAY  = 60             # 1分後に再エントリー
CYCLE_RESET_S  = 60 * 60        # 1.5時間で一旦リセット
SL_PCT         = 0.02           # ±3% ストップレンジ
RECV_WINDOW    = 50000
USER_AGENT     = "FundingArbBot/1.0"

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

    def open_basket(self):
        # funding取得
        fr_a = self.aster.get_funding_rate(SYMBOL)
        fr_b = self.bybit.get_funding_rate(SYMBOL)
        print(f"[funding] Aster={fr_a:.6f}, Bybit={fr_b:.6f}")

        # マーク価格
        mp_a = self.aster.get_mark_price(SYMBOL)
        mp_b = self.bybit.get_mark_price(SYMBOL)

        # サイド決定（Aster側, Bybit側）
        side_a, side_b = self._decide_sides(fr_a, fr_b)
        qty_a = self._compute_qty(mp_a)
        qty_b = self._compute_qty(mp_b)

        # エントリー
        leg_a = self._place_leg("ASTER", side_a, qty_a, mp_a)
        leg_b = self._place_leg("BYBIT", side_b, qty_b, mp_b)
        self.legs = (leg_a, leg_b)
        print(f"[enter] Aster {side_a} {qty_a} @~{mp_a:.2f}, SL~{leg_a.sl_price:.2f}")
        print(f"[enter] Bybit {side_b} {qty_b} @~{mp_b:.2f}, SL~{leg_b.sl_price:.2f}")

    def close_all(self, reason: str):
        print(f"[close_all] reason={reason}")
        # reduce-only で反対成行
        try:
            a_qty, _, b_qty, _ = self._get_positions()
            if self._is_open(a_qty):
                side = "SELL" if a_qty > 0 else "BUY"
                self.aster.place_market(SYMBOL, side, abs(a_qty), reduce_only=True)
            self.aster.cancel_all(SYMBOL)
        except Exception as e:
            print("[Aster close] err:", e)

        try:
            a_qty, _, b_qty, _ = self._get_positions()
            if self._is_open(b_qty):
                side = "SELL" if b_qty > 0 else "BUY"
                self.bybit.place_market(SYMBOL, side, abs(b_qty), reduce_only=True)
            self.bybit.cancel_all(SYMBOL)
        except Exception as e:
            print("[Bybit close] err:", e)

        self.legs = (None, None)

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

    def run(self):
        # 初回エントリー
        self.open_basket()
        last_reset = time.time()

        while True:
            time.sleep(POLL_SEC)

            # 1.5時間でリセット？
            if self.check_cycle_reset():
                self.close_all("cycle_reset_1.5h")
                self.open_basket()
                last_reset = time.time()
                continue

            action = self.poll_once()
            if action == "REENTER":
                print("[reenter] one side closed, syncing hedge...")
                # 両方クローズしてから再エントリー
                self.close_all("hedge_sync")
                time.sleep(REENTER_DELAY)
                self.open_basket()

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
    bot.run()

if __name__ == "__main__":
    main()


# %%



