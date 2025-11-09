# strategy.py
#Este archivo es tu “cerebro” del trader:decide si comprar, vender o quedarse quieto, según el precio justo.

from typing import Literal, Tuple, Optional

class Strategy:
    """
    Estrategia simple de Liquidity Taker:
    - Compra si el precio justo (p_fair) es menor a 0.45
    - Vende si el precio justo es mayor a 0.55
    - No hace nada si está entre esos valores
    """

    def entry_condition(
        self,
        p_fair: float,
        spread: float = 0.02
    ) -> Tuple[Optional[Literal["buy", "sell"]], float]:
        size = 1.0  # cantidad fija pequeña

        if p_fair < 0.45:
            return "buy", size
        elif p_fair > 0.55:
            return "sell", size
        else:
            return None, 0.0
