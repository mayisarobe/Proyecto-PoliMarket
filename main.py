# main.py

from fair_price import FairPrice
from strategy import Strategy

# Simulación de un orderbook de ejemplo
orderbook_mock = {
    "bids": [{"price": 0.40}],
    "asks": [{"price": 0.60}]
}

def main():
    print("🚀 Iniciando simulación de Polymarket Bot...")

    # 1️⃣ Calcular precio justo
    fp = FairPrice()
    p_fair = fp.mid_price(orderbook_mock)
    print(f"💰 Precio justo calculado: {p_fair}")

    # 2️⃣ Evaluar estrategia
    strat = Strategy()
    side, size = strat.entry_condition(p_fair)

    # 3️⃣ Resultado de la decisión
    if side:
        print(f"🧠 Decisión: {side.upper()} {size} unidades al precio {p_fair}")
    else:
        print("⏸️ No se cumplen condiciones de entrada al mercado.")

if __name__ == "__main__":
    main()
