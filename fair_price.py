# fair_price.py
#Este archivo es como una calculadora que saca el “precio justo” del mercado, promediando el precio de compra y el de venta más cercanos.


class FairPrice:
    """
    Calcula un precio justo (fair price) a partir del orderbook.
    Versión simple: usa el promedio entre el mejor bid y el mejor ask.
    """

    def mid_price(self, orderbook: dict) -> float:
        try:
            best_bid = float(orderbook["bids"][0]["price"])
            best_ask = float(orderbook["asks"][0]["price"])
            fair = (best_bid + best_ask) / 2
            return round(fair, 4)
        except Exception:
            # Si algo falla (por ejemplo, el orderbook está vacío)
            return 0.0
