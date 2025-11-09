# src/spread_calculator.py
from __future__ import annotations
import statistics
from typing import List, Tuple

class SpreadCalculator:
    """
    Avellaneda–Stoikov Spread Calculator
    ------------------------------------
    Calculates bid and ask quotes around a fair price using the Avellaneda–Stoikov model.
    It adjusts the reservation price (center) based on inventory to reduce risk and
    sets the spread width according to estimated volatility (sigma).

    Parameters
    ----------
    gamma : float
        Risk aversion parameter (↑ => wider spreads).
    lam : float
        Order arrival intensity proxy (↑ => tighter spreads).
    window : int
        Rolling window length to estimate sigma (volatility) from recent fair prices.
    min_spread : float
        Minimum full spread allowed (in probability space [0,1]).
    inv_skew_scale : float
        Scale factor for how much inventory shifts the quote center.

    Example
    -------
        sc = SpreadCalculator(gamma=0.15, lam=0.6, window=50, min_spread=0.003)
        bid, ask, full_spread = sc.quotes(fair=0.52, fair_var=1e-4, inv=5.0, inv_limit=30.0)
    """

    def __init__(
        self,
        gamma: float = 0.15,
        lam: float = 0.6,
        window: int = 50,
        min_spread: float = 0.003,
        inv_skew_scale: float = 1.0,
    ):
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.window = int(window)
        self.min_spread = float(min_spread)
        self.inv_skew_scale = float(inv_skew_scale)
        self._recent_fairs: List[float] = []

    # ---------- internal helpers ----------
    @staticmethod
    def _clip01(x: float) -> float:
        """Keep prices inside [0,1] since we're working with probabilities."""
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def _rolling_sigma(self) -> float:
        """Estimate rolling volatility (sigma) from recent fair prices."""
        if len(self._recent_fairs) < self.window:
            return 0.02  # conservative default
        return statistics.pstdev(self._recent_fairs[-self.window:]) + 1e-9

    def _half_spread(self, sigma: float) -> float:
        """Half-spread ≈ (γ * σ²) / (2λ)."""
        return (self.gamma * sigma * sigma) / max(1e-6, 2.0 * self.lam)

    def _reservation_price(self, fair: float, sigma: float, inv: float, inv_limit: float) -> float:
        """
        Reservation price r = fair - φ * γ * σ² * (inv / inv_limit)
        Shifts the center to offset inventory risk.
        """
        skew = self.inv_skew_scale * self.gamma * sigma * sigma * (inv / max(1.0, inv_limit))
        return fair - skew

    # ---------- public API ----------
    def quotes(self, fair: float, fair_var: float, inv: float, inv_limit: float) -> Tuple[float, float, float]:
        """
        Compute (bid, ask, full_spread) using the Avellaneda–Stoikov model.
        fair_var is included for compatibility (not directly used).
        """
        # Update history and estimate volatility
        self._recent_fairs.append(float(fair))
        sigma = self._rolling_sigma()

        # Compute reservation price and half spread
        r = self._reservation_price(fair, sigma, inv, inv_limit)
        half = self._half_spread(sigma)

        bid = self._clip01(r - half)
        ask = self._clip01(r + half)
        full = ask - bid

        # Enforce minimum spread if too tight
        if full < self.min_spread:
            half = self.min_spread / 2.0
            bid = self._clip01(r - half)
            ask = self._clip01(r + half)
            full = ask - bid

        return bid, ask, full


# --- minimal demo ---
if __name__ == "__main__":
    sc = SpreadCalculator(gamma=0.15, lam=0.6, window=50, min_spread=0.003)
    fair_series = [0.50, 0.51, 0.49, 0.52, 0.51, 0.50]
    inv, inv_limit = 5.0, 30.0

    for i, f in enumerate(fair_series):
        bid, ask, full = sc.quotes(fair=f, fair_var=1e-4, inv=inv, inv_limit=inv_limit)
        print(f"t={i:02d} fair={f:.3f} | bid={bid:.3f} ask={ask:.3f} spread={full:.4f}")
