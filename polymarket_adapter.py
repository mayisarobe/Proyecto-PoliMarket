# src/polymarket_adapter.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Iterable, Optional, List
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

@dataclass
class Tick:
    t: int
    mid: float
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None

@dataclass
class Fill:
    t: int
    side: str  # "buy" (compras tú) o "sell"
    price: float
    qty: float

class PolymarketAPIAdapter:
    """
    Adaptador entre el ClobClient (de PolymarketClient.c) y tu Market Maker.
    Ofrece una API sencilla: stream(), place_order(), cancel_all(), poll_fills(), get_inventory().
    """

    def __init__(
        self,
        client,
        market_id: str,
        yes_token: str,
        poll_sec: float = 1.0,
    ):
        self.c = client            # ClobClient autenticado (viene de tu PolymarketClient)
        self.market_id = market_id # id del mercado (ej. Champions)
        self.yes_token = yes_token # token del outcome YES
        self.poll_sec = float(poll_sec)
        self._t = 0

    # --- helpers ---
    @staticmethod
    def _mid_from_orderbook(ob: dict) -> tuple[float, Optional[float], Optional[float]]:
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        def first_price(side):
            if not side: return None
            v = side[0]
            if isinstance(v, (list, tuple)): return float(v[0])
            if isinstance(v, dict): return float(v.get("price", 0))
            return None
        bb = first_price(bids)
        ba = first_price(asks)
        if bb is None and ba is None: return 0.5, None, None
        if bb is None: return ba, None, ba
        if ba is None: return bb, bb, None
        return (bb + ba) / 2.0, bb, ba

    # --- métodos principales ---
    def stream(self) -> Iterable[Tick]:
        """Devuelve un stream de precios (mid) desde el orderbook."""
        while True:
            try:
                ob = self.c.get_orderbook(market=self.market_id, token_id=self.yes_token)
                mid, bb, ba = self._mid_from_orderbook(ob or {})
                self._t += 1
                yield Tick(t=self._t, mid=max(0.0, min(1.0, mid)), best_bid=bb, best_ask=ba)
            except Exception as e:
                print(f"[WARN] Error en stream(): {e}")
            time.sleep(self.poll_sec)

    def place_order(self, side: str, price: float, qty: float) -> str:
        """Envía orden LIMIT (buy/sell) a Polymarket."""
        side_const = BUY if side.lower() == "buy" else SELL
        args = OrderArgs(
            market=self.market_id,
            token=self.yes_token,
            side=side_const,
            price=float(price),
            size=float(qty),
            type=OrderType.LIMIT,
        )
        try:
            order = self.c.create_order(args)
            print(f"[OK] Orden {side.upper()} {qty}@{price:.3f}")
            return str(order.get("order_id", ""))
        except Exception as e:
            print(f"[ERROR] place_order(): {e}")
            return "error"

    def cancel_all(self) -> None:
        """Cancela todas las órdenes vivas."""
        try:
            self.c.cancel_all_orders()
            print("[OK] Órdenes canceladas")
        except Exception as e:
            print(f"[WARN] cancel_all(): {e}")

    def poll_fills(self) -> List[Fill]:
        """Consulta las ejecuciones (trades) recientes del usuario."""
        fills: List[Fill] = []
        try:
            trades = self.c.get_user_trades(market=self.market_id, token_id=self.yes_token)
            for tr in trades or []:
                side = tr.get("side", "").lower()
                price = float(tr.get("price", 0.0))
                size = float(tr.get("size", 0.0))
                fills.append(Fill(t=self._t, side=side, price=price, qty=size))
        except Exception as e:
            print(f"[WARN] poll_fills(): {e}")
        return fills

    def get_inventory(self) -> float:
        """Devuelve la posición neta en el token YES."""
        try:
            positions = self.c.get_positions()
            for p in positions or []:
                if str(p.get("token_id")) == str(self.yes_token):
                    return float(p.get("position", 0.0))
        except Exception as e:
            print(f"[WARN] get_inventory(): {e}")
        return 0.0
        