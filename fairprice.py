# fairprice.py
from __future__ import annotations
from typing import Optional, Iterable, Tuple
import math
import time


def clip01(x: float) -> float:
    """Recorta en [0,1]. Úsalo solo si trabajas en probabilidades (0–1)."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def ob_imbalance(bid_size: float, ask_size: float) -> float:
    """(B - A) / (B + A) ∈ [-1,1]."""
    den = max(1e-12, bid_size + ask_size)
    return (bid_size - ask_size) / den


class Ewma:
    def __init__(self, alpha: float, init: float = 0.0):
        self.alpha = float(alpha)
        self.state = float(init)
        self.ready = False

    def update(self, x: float) -> float:
        if not self.ready:
            self.state = float(x)
            self.ready = True
        else:
            self.state = self.alpha * float(x) + (1 - self.alpha) * self.state
        return self.state


class FairPrice:
    """
    Filtro de Kalman 1D con Q y R adaptativos + gating de outliers.

    IMPORTANTE:
    - Si clip_prob=True, asume precios en [0,1].
      Si usas precios en [0,100] desactiva clip_prob y ajusta Q/R.

    Métodos:
      - update(z) -> (x, P)
      - predict() -> x
      - update_from_orderbook(bid, ask, bid_sz=1.0, ask_sz=1.0, trade_lambda=None)
      - update_series(observations)
    """

    def __init__(
        self,
        initial_price: float = 0.50,
        initial_variance: float = 1.0,      # P0
        process_variance: float = 1e-4,     # Q base
        measurement_variance: float = 5e-3, # R base un poco mayor (menos paranoico)
        clip_prob: bool = True,
        eps: float = 1e-12,

        # ---- Adaptativos ----
        # R = max(r_min, (r_k * spread)^2) * (1 + r_imb_k * |imb|)
        r_k: float = 0.8,
        r_min: float = 1e-5,                # que no baje tanto (evita z-scores gigantes)
        r_imb_k: float = 0.3,               # sensibilidad a desequilibrio libro

        # Q = Q_base + q_gain * (EWMA(|Δmid|))^2 * (1 + q_lambda_k * λ_trades)
        q_alpha: float = 0.2,               # suavizado de micro-vol
        q_gain: float = 0.2,
        q_lambda_k: float = 0.2,            # opcional: impacto de intensidad de trades

        # ---- Outlier gating ----
        innovation_zscore_max: float = 8.0,  # umbral más laxo (antes 4.0)

        # ---- Warm-up ----
        warmup_steps: int = 5               # durante warmup no gatea
    ):
        # estado KF
        self.x = float(initial_price)
        self.P = float(initial_variance)
        self.Q0 = float(process_variance)
        self.R0 = float(measurement_variance)
        self.Q = self.Q0
        self.R = self.R0

        # config
        self.clip_prob = bool(clip_prob)
        self.eps = float(eps)

        # adaptativos
        self.r_k = float(r_k)
        self.r_min = float(r_min)
        self.r_imb_k = float(r_imb_k)
        self.q_gain = float(q_gain)
        self.q_lambda_k = float(q_lambda_k)

        self._ewma_abs_dmid = Ewma(alpha=q_alpha, init=0.0)
        self._last_mid: Optional[float] = None

        # gating y warmup
        self.zmax = float(innovation_zscore_max)
        self.warmup_steps = int(warmup_steps)
        self._steps = 0

        # métricas simples
        self.last_ts = time.time()
        self.innov_var = self.R0 + self.P + self.eps  # inicial

    # ----------------- utilidades internas -----------------
    def _maybe_clip(self, x: float) -> float:
        return clip01(x) if self.clip_prob else float(x)

    def _adapt_R(self, spread: float, imb_abs: float) -> float:
        spread_pos = max(0.0, spread)
        r_spread = max(self.r_min, (self.r_k * spread_pos) ** 2)
        return r_spread * (1.0 + self.r_imb_k * imb_abs)

    def _adapt_Q(self, mid: float, trade_lambda: Optional[float]) -> float:
        # micro-vol: |Δmid|
        if self._last_mid is None:
            micro = 0.0
        else:
            micro = abs(mid - self._last_mid)
        vol = self._ewma_abs_dmid.update(micro)
        q = self.Q0 + self.q_gain * (vol ** 2)
        if trade_lambda is not None:  # λ en [0,∞), normalizado si quieres
            q *= (1.0 + self.q_lambda_k * float(trade_lambda))
        self._last_mid = mid
        return q

    # ----------------- API pública -----------------
    def update(self, z: float) -> Tuple[float, float]:
        """Ingiere observación z (precio) y retorna (x, P)."""
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Innovation
        y = float(z) - x_pred
        S = P_pred + self.R + self.eps  # var de la innovación
        self.innov_var = S

        # Gating (evita saltos absurdos salvo en warmup)
        if self._steps > self.warmup_steps:
            denom = math.sqrt(max(self.eps, S))
            zscore = abs(y) / denom
            if zscore > self.zmax:
                # rechaza medición: solo propaga incertidumbre
                self.x = x_pred
                self.P = P_pred
                self._steps += 1
                return self._maybe_clip(self.x), self.P

        # Update KF
        K = P_pred / S
        self.x = x_pred + K * y
        self.P = (1.0 - K) * P_pred

        self.x = self._maybe_clip(self.x)
        self._steps += 1
        return self.x, self.P

    def predict(self) -> float:
        """Paso sin observación."""
        self.P += self.Q
        return self._maybe_clip(self.x)

    def set_noise(
        self,
        process_variance: Optional[float] = None,
        measurement_variance: Optional[float] = None,
    ) -> None:
        """Actualizar Q0 y/o R0 (y valores actuales Q,R)."""
        if process_variance is not None:
            self.Q0 = float(process_variance)
            self.Q = self.Q0
        if measurement_variance is not None:
            self.R0 = float(measurement_variance)
            self.R = self.R0

    def reset(
        self,
        initial_price: float = 0.50,
        initial_variance: float = 1.0,
    ) -> None:
        """Reinicia el filtro por completo."""
        self.x = float(initial_price)
        self.P = float(initial_variance)
        self._last_mid = None
        self._ewma_abs_dmid = Ewma(alpha=self._ewma_abs_dmid.alpha, init=0.0)
        self._steps = 0
        # importante: devolver Q y R a sus bases
        self.Q = self.Q0
        self.R = self.R0
        self.innov_var = self.R0 + self.P + self.eps
        self.last_ts = time.time()

    def update_from_orderbook(
        self,
        best_bid: float,
        best_ask: float,
        bid_size: float = 1.0,
        ask_size: float = 1.0,
        trade_lambda: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Convierte el libro en mid/spread y aplica Q,R adaptativos:
          R = f(spread, imbalance)
          Q = f(microvol, λ)

        NOTA:
        - Asume que best_bid/best_ask ya están en la escala correcta:
          * 0–1 si vas a usar clip_prob=True (probabilidades estilo Polymarket).
          * 0–100 o lo que sea si clip_prob=False y has ajustado Q/R.
        """
        b = float(best_bid)
        a = float(best_ask)
        if a < b:  # arregla libro roto
            a = b

        mid = 0.5 * (a + b)
        spread = max(0.0, a - b)
        imb = abs(ob_imbalance(bid_size, ask_size))

        # adaptar ruidos para este tick (sin pisar los base)
        Q_prev, R_prev = self.Q, self.R
        try:
            self.Q = self._adapt_Q(mid, trade_lambda)
            self.R = self._adapt_R(spread, imb)
            return self.update(mid)
        finally:
            self.Q, self.R = Q_prev, R_prev

    def update_series(
        self,
        mids: Iterable[float],
    ) -> Iterable[Tuple[float, float]]:
        """
        Batch simple sobre una serie de mids ya calculados.
        Usa Q adaptativo y R=R0 fija.
        """
        out: list[Tuple[float, float]] = []
        for z in mids:
            self.Q = self._adapt_Q(float(z), trade_lambda=None)
            self.R = self.R0
            out.append(self.update(float(z)))
        return out
