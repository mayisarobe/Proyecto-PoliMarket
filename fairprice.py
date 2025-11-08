from __future__ import annotations
from typing import Optional

class FairPrice:
    """
    Kalman 1D para precio justo.
    - Interfaz drop-in con tus parámetros.
    - Corrige: update retorna (x, P) y aplica clip01.
    - Opcional: update_from_orderbook() ajusta R con el spread y Q con una EWMA de |Δmid|.
    """

    def __init__(
        self,
        initial_price: float = 0.0,
        process_variance: float = 1e-5,       # Q base
        measurement_variance: float = 1e-2,   # R base
        initial_variance: float = 1.0,        # P0
        clip01: bool = False,
        eps: float = 1e-12
    ):
        self.x = float(initial_price)         # posterior mean
        self.P = float(initial_variance)      # posterior var
        self.Q = float(process_variance)
        self.R = float(measurement_variance)
        self.clip01 = bool(clip01)
        self.eps = float(eps)

        # Estado auxiliar para ruidos dinámicos (opcional)
        self._last_mid: Optional[float] = None
        self._ewma_abs_dmid: float = 0.0

        # Tuning por defecto para dinámicos
        self.r_k: float = 0.5     # escala para spread → R = max(r_min, (r_k*spread)^2)
        self.r_min: float = 1e-6
        self.q_alpha: float = 0.1 # EWMA para |Δmid|
        self.q_gain: float = 0.1  # cuánto contribuye la micro-vol al Q

    # ---------- util ----------
    def _maybe_clip(self, x: float) -> float:
        if not self.clip01: return x
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    # ---------- API principal ----------
    def update(self, z: float) -> tuple[float, float]:
        """
        Ingiere observación z y devuelve (fair_price, posterior_variance).
        Usa Q y R actuales (fijos o los que hayas seteado antes).
        """
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        denom = P_pred + self.R + self.eps
        K = P_pred / denom
        self.x = x_pred + K * (float(z) - x_pred)
        self.P = (1.0 - K) * P_pred

        # Clip si es probabilidad
        self.x = self._maybe_clip(self.x)
        return self.x, self.P

    def predict(self) -> float:
        """
        Paso sin observación: solo aumenta la incertidumbre.
        """
        self.P += self.Q
        return self.x

    def set_noise(self, process_variance: float | None = None, measurement_variance: float | None = None) -> None:
        """Ajusta Q y/o R manualmente."""
        if process_variance is not None:
            self.Q = float(process_variance)
        if measurement_variance is not None:
            self.R = float(measurement_variance)

    def reset(self, initial_price: float = 0.0, initial_variance: float = 1.0) -> None:
        """Resetea estado y covarianza."""
        self.x = float(initial_price)
        self.P = float(initial_variance)
        self._last_mid = None
        self._ewma_abs_dmid = 0.0

    # ---------- Azúcar para usar con orderbook (opcional) ----------
    def update_from_orderbook(self, best_bid: float, best_ask: float) -> tuple[float, float]:
        """
        Construye z = mid y aplica ruidos dinámicos:
          - R = max(r_min, (r_k * spread)^2)
          - Q = Q_base + q_gain * (EWMA(|Δmid|))^2
        Devuelve (x, P).
        """
        b = float(best_bid)
        a = float(best_ask)
        if a < b:  # libro roto → forzar no negativo
            a = b
        mid = 0.5 * (a + b)
        spread = max(a - b, 0.0)

        # Actualiza EWMA de micro-vol
        if self._last_mid is None:
            self._ewma_abs_dmid = 0.0
        else:
            d = abs(mid - self._last_mid)
            self._ewma_abs_dmid = (1 - self.q_alpha) * self._ewma_abs_dmid + self.q_alpha * d
        self._last_mid = mid

        # Ajustes dinámicos
        dyn_R = max(self.r_min, (self.r_k * spread) ** 2)
        dyn_Q = self.Q + self.q_gain * (self._ewma_abs_dmid ** 2)

        # Guardar temporalmente (si prefieres no pisar Q/R globales)
        Q_prev, R_prev = self.Q, self.R
        self.Q, self.R = dyn_Q, dyn_R
        try:
            return self.update(mid)
        finally:
            self.Q, self.R = Q_prev, R_prev
