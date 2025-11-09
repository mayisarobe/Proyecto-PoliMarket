# spread.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class SpreadCalculator:
    """
    Calculadora de spreads (Avellaneda–Stoikov “lite”) adaptada a Polymarket.

    - Usa una señal de microestructura vía EWMA de |Δmid| (update_micro).
    - Mezcla esa señal con la varianza del fair (fair_var) para estimar sigma.
    - Aplica un spread base dependiente de sigma, con piso (min_spread)
      y un suavizado hacia los bordes (edge_k).
    - Desplaza el centro (reservation price) por inventario (inv / inv_limit),
      escalado por inv_skew_scale * k_imb.

    Parámetros principales
    ----------------------
    gamma : float
        Aversión al riesgo (se usa como “peso” implícito en el spread).
    lam : float
        Intensidad de llegada de órdenes (no se usa de forma explícita en esta
        versión “lite”, pero queda para extensiones si lo necesitás).
    window : int
        Longitud de ventana lógica para el tracking de microestructura.
    ewma_alpha : float
        Factor de suavizado para EWMA(|Δmid|). [0,1], más alto = reactividad mayor.
    k_spread : float
        Ganancia para mapear sigma → spread.
    k_imb : float
        Ganancia del sesgo por inventario.
    edge_k : float
        Aumento de spread cerca de los bordes (0 o 1), en función de min(fair, 1-fair).
    hard_max : float
        Techo duro del spread (por seguridad).
    min_spread : float
        Piso mínimo del spread (prob space [0,1]).
    inv_skew_scale : float
        Escala adicional del sesgo por inventario (permite “afilar” o “suavizar” el skew).
    """

    def __init__(
        self,
        gamma: float,
        lam: float,
        window: int = 100,
        ewma_alpha: float = 0.20,
        k_spread: float = 0.35,
        k_imb: float = 0.015,
        edge_k: float = 0.30,
        hard_max: float = 0.20,
        min_spread: float = 0.003,
        inv_skew_scale: float = 1.0,
    ) -> None:
        # Hiperparámetros
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.window = int(window)
        self.ewma_alpha = float(ewma_alpha)
        self.k_spread = float(k_spread)
        self.k_imb = float(k_imb)
        self.edge_k = float(edge_k)
        self.hard_max = float(hard_max)
        self.min_spread = float(min_spread)
        self.inv_skew_scale = float(inv_skew_scale)

        # Estado interno (microestructura)
        self._last_mid: Optional[float] = None
        self._ewma_abs_dmid: float = 0.0  # EWMA de |Δmid|
        self._mids: List[float] = []      # histórico corto, por si querés extender

    # ---------------- utils ----------------
    @staticmethod
    def _clip01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    # ---------------- API micro ----------------
    def update_micro(self, best_bid: float, best_ask: float) -> None:
        """
        Actualiza señales de microestructura a partir del mejor bid/ask.
        - mid = (bid + ask) / 2
        - EWMA(|Δmid|) como proxy de micro-volatilidad intradía.
        """
        bid = float(best_bid)
        ask = float(best_ask)
        if ask < bid:
            ask = bid

        mid = 0.5 * (bid + ask)

        if self._last_mid is None:
            self._ewma_abs_dmid = 0.0
        else:
            d = abs(mid - self._last_mid)
            a = self.ewma_alpha
            self._ewma_abs_dmid = (1.0 - a) * self._ewma_abs_dmid + a * d

        self._last_mid = mid
        self._mids.append(mid)
        if len(self._mids) > self.window:
            self._mids.pop(0)

    # ---------------- API principal ----------------
    def quotes(
        self,
        fair: float,
        fair_var: float,
        inv: float = 0.0,
        inv_limit: float = 30.0,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Devuelve (bid, ask, meta) donde meta contiene:
            - spread: spread final aplicado
            - skew  : desplazamiento por inventario
            - sigma : volatilidad efectiva usada
        """
        fair = float(fair)
        fair_var = float(fair_var)
        inv = float(inv)
        inv_limit = max(float(inv_limit), 1e-6)

        # 1) Sigma efectiva: mezcla fair_var y micro EWMA(|Δmid|)
        sigma_from_fair = fair_var ** 0.5 if fair_var > 0.0 else 0.0
        sigma_micro = self._ewma_abs_dmid
        sigma = max(sigma_from_fair, sigma_micro, 1e-8)  # evitá cero

        # 2) Spread base con borde y límites
        edge_boost = self.edge_k * min(fair, 1.0 - fair)  # +spread cerca de 0/1
        spread = self.k_spread * sigma + edge_boost
        spread = max(self.min_spread, min(self.hard_max, spread))

        # 3) Sesgo por inventario (reservation price shift)
        skew = self.inv_skew_scale * self.k_imb * (inv / inv_limit)
        r = fair - skew

        # 4) Cotizaciones
        half = 0.5 * spread
        bid = self._clip01(r - half)
        ask = self._clip01(r + half)

        # Garantía de min_spread tras clip
        if ask - bid < self.min_spread:
            half = 0.5 * self.min_spread
            bid = self._clip01(r - half)
            ask = self._clip01(r + half)

        # Meta útil para debugging/reportes
        meta = {"spread": float(ask - bid), "skew": float(skew), "sigma": float(sigma)}
        return bid, ask, meta

