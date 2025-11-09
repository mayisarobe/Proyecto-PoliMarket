# spread.py
from __future__ import annotations
from typing import List, Tuple, Optional
import math

class _Ewma:
    def __init__(self, alpha: float = 0.2, init: Optional[float] = None):
        self.alpha = float(alpha)
        self.state = init
    def update(self, x: float) -> float:
        x = float(x)
        self.state = x if self.state is None else self.alpha * x + (1 - self.alpha) * self.state
        return self.state if self.state is not None else x

class SpreadCalculator:
    """
    Avellaneda–Stoikov mejorado (prob space [0,1]) con señales de microestructura.

    Parámetros claves
    -----------------
    gamma: aversión al riesgo (↑ → spreads más anchos)
    lam: intensidad de llegada de órdenes (↑ → spreads más estrechos)
    window: tamaño de ventana lógica (se usa vía EWMA; no hace falta exactamente)
    min_spread: piso de spread total
    inv_skew_scale: escala del sesgo por inventario
    ewma_alpha: suavizado de retornos para sigma
    k_spread: cuánto contribuye el spread del libro al spread cotizado
    k_imb: cuánto contribuye el imbalance al spread cotizado
    edge_k: ensancha spreads cerca de 0/1
    hard_max: tope de spread para no alejarse demasiado
    """

    def __init__(
        self,
        gamma: float = 0.15,
        lam: float = 0.6,
        window: int = 50,
        min_spread: float = 0.003,
        inv_skew_scale: float = 1.0,
        ewma_alpha: float = 0.25,
        k_spread: float = 0.30,
        k_imb: float = 0.01,
        edge_k: float = 0.25,
        hard_max: float = 0.15,
    ):
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.window = int(window)
        self.min_spread = float(min_spread)
        self.inv_skew_scale = float(inv_skew_scale)
        self.ewma_alpha = float(ewma_alpha)
        self.k_spread = float(k_spread)
        self.k_imb = float(k_imb)
        self.edge_k = float(edge_k)
        self.hard_max = float(hard_max)

        # estado
        self._fair_prev: Optional[float] = None
        self._r_ewma = _Ewma(alpha=self.ewma_alpha, init=0.0)  # EWMA de retornos (dfair)
        self._last_book_spread: float = 0.0
        self._last_imb: float = 0.0  # |imb| en [0,1]

        # buffer de fairs por compatibilidad (no lo usamos duro)
        self._recent_fairs: List[float] = []

    # ---------- helpers ----------
    @staticmethod
    def _clip01(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    @staticmethod
    def _imbalance(bid_sz: float, ask_sz: float) -> float:
        den = max(1e-9, bid_sz + ask_sz)
        return (bid_sz - ask_sz) / den  # [-1, 1]

    def update_micro(self, best_bid: Optional[float], best_ask: Optional[float],
                     bid_size: Optional[float] = None, ask_size: Optional[float] = None) -> None:
        """
        Llamá a esto por tick si querés que el spread refleje el libro:
        - Guarda el último spread del libro (ask-bid).
        - Guarda |imbalance| de tamaños (opcional).
        """
        if best_bid is not None and best_ask is not None:
            self._last_book_spread = max(0.0, float(best_ask) - float(best_bid))
        if bid_size is not None and ask_size is not None:
            self._last_imb = abs(self._imbalance(float(bid_size), float(ask_size)))

    def _sigma_from_fair(self, fair: float) -> float:
        # retorno simple dfair
        if self._fair_prev is None:
            self._fair_prev = float(fair)
            r = 0.0
        else:
            r = float(fair) - float(self._fair_prev)
            self._fair_prev = float(fair)
        # EWMA de |retorno| como proxy de sigma (rápido, robusto)
        sig = self._r_ewma.update(abs(r))
        # piso conservador
        return max(sig, 0.005)

    def _half_spread_base(self, sigma: float) -> float:
        # fórmula Avellaneda–Stoikov aproximada
        return (self.gamma * sigma * sigma) / max(1e-6, 2.0 * self.lam)

    def _reservation_price(self, fair: float, sigma: float, inv: float, inv_limit: float) -> float:
        # sesgo con tanh para no desbocar en extremos
        ratio = float(inv) / max(1.0, float(inv_limit))
        skew = self.inv_skew_scale * self.gamma * sigma * sigma * math.tanh(3.0 * ratio)
        return fair - skew

    # ---------- API pública ----------
    def quotes(self, fair: float, fair_var: float, inv: float, inv_limit: float) -> Tuple[float, float, float]:
        """
        Devuelve (bid, ask, full_spread). Compatible con tu main.py.
        Si no llamaste update_micro(), usa valores neutros (spread libro=0, |imb|=0).
        """
        fair = float(fair)
        self._recent_fairs.append(fair)

        # 1) sigma por EWMA de retornos del fair
        sigma = self._sigma_from_fair(fair)

        # 2) base half-spread Avellaneda–Stoikov
        half = self._half_spread_base(sigma)

        # 3) microestructura: añadimos contribuciones
        half += 0.5 * ( self.k_spread * self._last_book_spread + self.k_imb * self._last_imb )

        # 4) widen near edges (cerca de 0/1)
        edge_factor = 1.0 + self.edge_k * (1.0 - 2.0 * abs(fair - 0.5))  # ~1.25 en 0/1, ~1.0 en 0.5
        half *= edge_factor

        # 5) reservation price con inventario (tanh)
        r = self._reservation_price(fair, sigma, inv, inv_limit)

        # 6) cotizaciones y guardas
        bid = self._clip01(r - half)
        ask = self._clip01(r + half)
        full = max(0.0, ask - bid)

        # 7) mínimos y máximos
        if full < self.min_spread:
            half = self.min_spread / 2.0
            bid = self._clip01(r - half)
            ask = self._clip01(r + half)
            full = ask - bid

        full = min(full, self.hard_max)

        # 8) anticrossing (por si el clip01 generó cruce)
        if bid >= ask:
            mid = (bid + ask) / 2.0
            bid = self._clip01(mid - self.min_spread / 2.0)
            ask = self._clip01(mid + self.min_spread / 2.0)
            full = ask - bid

        return bid, ask, full


# --- ejemplo mínimo ---
if __name__ == "__main__":
    sc = SpreadCalculator(gamma=0.15, lam=0.6, window=50, min_spread=0.003)
    fair_series = [0.50, 0.51, 0.49, 0.52, 0.51, 0.50]
    inv, inv_limit = 5.0, 30.0
    for i, f in enumerate(fair_series):
        # ejemplo de microestructura opcional
        sc.update_micro(best_bid=0.49, best_ask=0.51, bid_size=100, ask_size=120)
        bid, ask, full = sc.quotes(fair=f, fair_var=1e-4, inv=inv, inv_limit=inv_limit)
        print(f"t={i:02d} fair={f:.3f} | bid={bid:.3f} ask={ask:.3f} spread={full:.4f}")
