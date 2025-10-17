# pruebas.py
import os
from dotenv import load_dotenv
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY
from client import PolymarketClient  

# fuerza que cargue el .env correcto
env_path = os.path.join(os.path.dirname(__file__), ".env")
print("Usando .env:", env_path)
load_dotenv(dotenv_path=env_path, override=True)

def ok(msg): print(f"✅ {msg}")
def fail(msg): print(f"❌ {msg}")

def main():
    mode = os.getenv("LOGIN_MODE", "wallet")
    print(f"[INFO] LOGIN_MODE={mode}")

    # 1️⃣ crear cliente
    try:
        cli = PolymarketClient()
        ok("Cliente inicializado correctamente")
    except Exception as e:
        fail(f"No se pudo inicializar el cliente: {e}")
        return

    # 2️⃣ endpoint público /markets
    try:
        mkts = cli.c.get_markets()
        total = len(mkts["data"]) if isinstance(mkts, dict) and "data" in mkts else len(mkts)
        ok(f"/markets OK → {total} mercados devueltos")
    except Exception as e:
        fail(f"Error en /markets: {e}")
        return

    # 3️⃣ endpoint privado (autenticado)
    try:
        if hasattr(cli.c, "get_orders"):
            orders = cli.c.get_orders()  
            ok(f"get_orders OK → {orders}")
        else:
            try:
                cli.c.cancel_order("00000000-0000-0000-0000-000000000000")
            except Exception as e:
                ok(f"ruta privada responde (cancel fake) → {e}")
    except Exception as e:
        fail(f"Fallo en ruta privada: {e}")
        return

    # 4️⃣ firmar orden localmente (sin enviarla)
    try:
        demo_token = "114304586861386186441621124384163963092522056897081085884483958561365015034812"
        signed = cli.c.create_order(OrderArgs(price=0.01, size=5.0, side=BUY, token_id=demo_token))
        ok("create_order OK → orden firmada")
    except Exception as e:
        fail(f"Error al firmar orden: {e}")
        return

    # 5️⃣ probar post_order (no gasta si la wallet está vacía)
    try:
        resp = cli.c.post_order(signed, OrderType.GTC)
        ok(f"post_order OK → {resp}")
    except Exception as e:
        ok(f"post_order respondió (esperado en pruebas): {e}")

if __name__ == "__main__":
    main()
