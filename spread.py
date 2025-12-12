# spread.py
from __future__ import annotations
import statistics
from typing import List, Tuple

class SpreadCalculator:
    """
    Avellaneda–Stoikov lite con:
    - sigma por EWMA (más memoriosa),
    - ancho mínimo por micro-spread del libro,
    - ajuste del centro por inventario.
    """

    def __init__(
        self,
        gamma: float = 0.15,
        lam: float = 0.6,
        window: int = 200,
        k_spread: float = 0.35,   # factor para transformar sigma en half-spread
        edge_k: float = 0.30,     # penaliza extremos (0/1)
        hard_max: float = 0.20,   # techo de half-spread
        ewma_alpha: float = 0.15, # memoria de sigma
        k_imb: float = 0.015,     # sesgo por inventario
    ):
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.window = int(window)
        self.k_spread = float(k_spread)
        self.edge_k = float(edge_k)
        self.hard_max = float(hard_max)
        self.ewma_alpha = float(ewma_alpha)
        self.k_imb = float(k_imb)

        self._recent_fairs: List[float] = []
        self._ewma_sigma: float | None = None
        self._micro_half_floor: float = 0.0  # piso dinámico por spread del libro

    # ---------- utils ----------
    @staticmethod
    def _clip01(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def _update_sigma(self, fair: float) -> float:
        """EWMA de retornos absolutos como proxy de sigma."""
        self._recent_fairs.append(float(fair))
        if len(self._recent_fairs) < 2:
            self._ewma_sigma = self._ewma_sigma or 0.02
            return self._ewma_sigma

        r = abs(self._recent_fairs[-1] - self._recent_fairs[-2])
        if self._ewma_sigma is None:
            self._ewma_sigma = r
        else:
            a = self.ewma_alpha
            self._ewma_sigma = (1 - a) * self._ewma_sigma + a * r
        return max(self._ewma_sigma, 1e-6)

    # ---------- microestructura ----------
    def update_micro(self, best_bid: float | None, best_ask: float | None) -> None:
        """Usa el spread real del libro como piso de half-spread."""
        if best_bid is None or best_ask is None:
            return
        spr = max(0.0, float(best_ask) - float(best_bid))
        # piso = la mitad del spread real del libro
        self._micro_half_floor = max(self._micro_half_floor * 0.8, spr * 0.5)

    # ---------- API principal ----------
    def quotes(self, fair: float, fair_var: float, inv: float, inv_limit: float) -> Tuple[float, float, float]:
        # 1) sigma y half-spread base
        sigma = self._update_sigma(fair)
        half = self.k_spread * sigma

        # 2) penalización en extremos (cuando fair ~0 o ~1)
        edge = self.edge_k * abs(0.5 - fair)
        half = max(half, edge)

        # 3) piso por microestructura (best-ask - best-bid) / 2
        half = max(half, self._micro_half_floor)

        # 4) techo duro
        half = min(half, self.hard_max)

        # 5) centro con sesgo por inventario
        skew = self.k_imb * (inv / max(1.0, inv_limit))
        r = fair - skew

        bid = self._clip01(r - half)
        ask = self._clip01(r + half)
        full = max(0.0, ask - bid)
        return bid, ask, full
