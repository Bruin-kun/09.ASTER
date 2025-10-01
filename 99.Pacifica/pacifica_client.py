# pacifica_client.py
# -*- coding: utf-8 -*-
import os
import time
import uuid
import pathlib
import requests
import base58
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from solders.keypair import Keypair

from common.constants import REST_URL
from common.utils import sign_message

# keys.env を明示読み込み
load_dotenv("keys.env")

OWNER_PRIVATE_KEY = os.getenv("OWNER_PRIVATE_KEY_BASE58")
API_KEY           = os.getenv("PACIFICA_API_KEY")
AGENT_KEY_PATH    = os.getenv("AGENT_KEY_PATH", "agent_wallet_base58.key")

# --- Endpoints ---
BIND_URL    = f"{REST_URL}/agent/bind"
MKT_URL     = f"{REST_URL}/orders/create_market"
LIMIT_URL   = f"{REST_URL}/orders/create"
CANCEL_URL  = f"{REST_URL}/orders/cancel"
CANCEL_ALL  = f"{REST_URL}/orders/cancel_all"


# =========================
# 共通ユーティリティ
# =========================
def _headers() -> Dict[str, str]:
    """共通ヘッダ（必要ならAPIキーも付与）"""
    h = {"Content-Type": "application/json"}
    if API_KEY:
        # 公式の指定ヘッダ名に合わせて必要なら変更
        h["X-API-Key"] = API_KEY
    return h


def _post(url: str, body: Dict[str, Any],
          max_retries: int = 3, base_delay: float = 0.5) -> requests.Response:
    """
    POST（429/5xx は指数バックオフでリトライ）
    """
    last: Optional[requests.Response] = None
    for i in range(max_retries):
        r = requests.post(url, json=body, headers=_headers(), timeout=15)
        # 2xx/4xx(429以外) は即返す。429と5xxのみリトライ
        if (r.status_code < 500) and (r.status_code != 429):
            return r
        last = r
        time.sleep(base_delay * (2 ** i))
    return last  # 最後のレスポンス


def _now_ms() -> int:
    return int(time.time() * 1_000)


def _coid(prefix: str = "ast") -> str:
    return f"{prefix}-{_now_ms()}-{uuid.uuid4().hex[:8]}"


# =========================
# 鍵まわり
# =========================
def load_owner() -> Keypair:
    """オーナー（メイン）鍵を .env から読み込み"""
    if not OWNER_PRIVATE_KEY:
        raise RuntimeError("OWNER_PRIVATE_KEY_BASE58 が keys.env にありません。")
    return Keypair.from_base58_string(OWNER_PRIVATE_KEY)


def load_or_create_agent(path: str = AGENT_KEY_PATH) -> Keypair:
    """
    エージェント鍵をファイルから読み込み。
    無ければ新規生成して base58 で保存（権限600にトライ）
    """
    p = pathlib.Path(path)
    if p.exists():
        return Keypair.from_base58_string(p.read_text().strip())
    k = Keypair()
    p.write_text(base58.b58encode(k.to_bytes()).decode())
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass
    return k


# =========================
# 署名付きボディ作成（共通）
# =========================
def _signed_request(owner_pub: str,
                    signer_kp: Keypair,
                    op_type: str,
                    payload: Dict[str, Any],
                    agent_pub: Optional[str] = None,
                    expiry_window_ms: int = 5_000) -> Dict[str, Any]:
    """
    Pacificaの署名仕様に従い、共通の signed body を作る
    """
    ts = _now_ms()
    header = {"timestamp": ts, "expiry_window": expiry_window_ms, "type": op_type}
    _, sig = sign_message(header, payload, signer_kp)

    body: Dict[str, Any] = {
        "account": owner_pub,
        "signature": sig,
        "timestamp": ts,
        "expiry_window": expiry_window_ms,
        **payload,
    }
    if agent_pub:
        body["agent_wallet"] = agent_pub
    return body


# =========================
# エージェント紐付け
# =========================
def bind_agent_wallet(owner_kp: Keypair, agent_pubkey: str) -> requests.Response:
    owner_pub = str(owner_kp.pubkey())
    payload = {"agent_wallet": agent_pubkey}
    body = _signed_request(owner_pub, owner_kp, "bind_agent_wallet", payload)
    return _post(BIND_URL, body)


# =========================
# 成行（TP/SL同梱対応）
# =========================
def create_market_order(owner_pub: str,
                        agent_kp: Keypair,
                        symbol: str,
                        side: str,               # "bid" or "ask"
                        amount: str,             # 文字列で送る（小数精度は呼び出し側で調整）
                        reduce_only: bool = False,
                        slippage_percent: str = "0.5",
                        client_order_id: Optional[str] = None,
                        take_profit: Optional[Dict[str, Any]] = None,
                        stop_loss: Optional[Dict[str, Any]] = None,
                        expiry_window_ms: int = 5_000) -> requests.Response:
    """
    Market Order（/orders/create_market）
    - TP/SL をネストで同梱可（例：{"stop_price":"48000","limit_price":"47950","client_order_id":"..."}）
    """
    if client_order_id is None:
        client_order_id = _coid("mkt")

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "reduce_only": reduce_only,
        "amount": amount,
        "side": side,
        "slippage_percent": slippage_percent,
        "client_order_id": client_order_id,
    }
    if take_profit is not None:
        payload["take_profit"] = take_profit
    if stop_loss is not None:
        payload["stop_loss"] = stop_loss

    body = _signed_request(
        owner_pub,
        agent_kp,
        "create_market_order",
        payload,
        agent_pub=str(agent_kp.pubkey()),
        expiry_window_ms=expiry_window_ms,
    )
    return _post(MKT_URL, body)


# =========================
# 指値（参考：据え置き）
# =========================
def create_limit_order(owner_pub: str,
                       agent_kp: Keypair,
                       symbol: str,
                       side: str,               # "bid"/"ask"
                       amount: str,
                       price: str,
                       reduce_only: bool = False,
                       client_order_id: Optional[str] = None,
                       take_profit: Optional[str] = None,
                       stop_loss: Optional[str] = None,
                       expiry_window_ms: int = 5_000) -> requests.Response:
    """
    Limit Order（/orders/create）
    - ここでは TP/SL を文字列（価格）として同梱する簡易形（必要に応じて dict 仕様に変更可）
    """
    if client_order_id is None:
        client_order_id = _coid("lmt")

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "amount": amount,
        "price": price,
        "reduce_only": reduce_only,
        "client_order_id": client_order_id,
    }
    if take_profit is not None:
        payload["take_profit"] = take_profit
    if stop_loss is not None:
        payload["stop_loss"] = stop_loss

    body = _signed_request(
        owner_pub,
        agent_kp,
        "create_order",
        payload,
        agent_pub=str(agent_kp.pubkey()),
        expiry_window_ms=expiry_window_ms,
    )
    return _post(LIMIT_URL, body)


# =========================
# キャンセル（単体）
# =========================
def cancel_order(owner_pub: str,
                 agent_kp: Keypair,
                 symbol: str,
                 order_id: Optional[int] = None,
                 client_order_id: Optional[str] = None,
                 expiry_window_ms: int = 5_000) -> requests.Response:
    """
    /orders/cancel
    - symbol は必須
    - order_id か client_order_id のどちらか必須
    """
    if (order_id is None) and (client_order_id is None):
        raise ValueError("order_id か client_order_id のいずれかを指定してください。")

    payload: Dict[str, Any] = {"symbol": symbol}
    if order_id is not None:
        payload["order_id"] = order_id
    if client_order_id is not None:
        payload["client_order_id"] = client_order_id

    body = _signed_request(
        owner_pub,
        agent_kp,
        "cancel_order",
        payload,
        agent_pub=str(agent_kp.pubkey()),
        expiry_window_ms=expiry_window_ms,
    )
    return _post(CANCEL_URL, body)


# =========================
# キャンセル（全て／銘柄指定）
# =========================
def cancel_all_orders(owner_pub: str,
                      agent_kp: Keypair,
                      symbols: Optional[List[str]] = None,
                      expiry_window_ms: int = 5_000) -> requests.Response:
    """
    /orders/cancel_all
    - 全銘柄 or symbols（配列）で取り消し
    """
    payload: Dict[str, Any] = {}
    if symbols:
        payload["symbols"] = symbols  # 仕様により "symbol" 単数の可能性があれば調整

    body = _signed_request(
        owner_pub,
        agent_kp,
        "cancel_all_orders",
        payload,
        agent_pub=str(agent_kp.pubkey()),
        expiry_window_ms=expiry_window_ms,
    )
    return _post(CANCEL_ALL, body)
