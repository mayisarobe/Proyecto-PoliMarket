# 💻 Market Making Algorítmico en Polymarket - Pitch a un Venture Capital

## 🖊️ Descripción
Desarrollo e implementación de un sistema de market making algorítmico orientado a operar en Polymarket. 

Este proyecto forma parte de la práctica final del curso.  
El objetivo es desarrollar un **algoritmo de market making o liquidity taker**
en el mercado de predicciones **Polymarket**, tomando como caso de estudio
el mercado del **First to 5k: Gold or ETH?**.

## 🧩 Estructura del Proyecto

- `client.py` → conexión con la API de Polymarket (CLOB client).
- `polymarket_adapter.py` → capa de abstracción para interactuar con Polymarket.
- `fairprice.py` → cálculo del fair price (baseline: mid-price).
- `spread.py` → lógica de cálculo del spread.
- `main.py` → orquestador principal (backtest / live).
- `fetch_trades_history.py` → descarga y gestión de histórico de trades.
- `.env` → variables sensibles y claves privadas (no se sube al repo).
- `.gitignore` → evita subir archivos sensibles o temporales.

## ⚙️ Instalación
```bash
pip install -r requirements.txt
```
## ▶️ Ejecución

El sistema puede ejecutarse en dos modos distintos en tiempo real:

- **Market Maker**: el algoritmo cotiza precios de compra y venta de forma continua.
- **Liquidity Taker**: el algoritmo solo ejecuta órdenes cuando se cumplen ciertas condiciones.

También incluye un modo de backtest para pruebas sobre histórico.

Backtest:
```bash
python main.py --token_id <TOKEN_ID> --mode backtest --samples <SAMPLES> --interval <INTERVAL> --debug
```

Live — Market Maker:
```bash
python main.py --token_id <TOKEN_ID> --mode live --role maker --order_size <ORDER_SIZE> --interval <INTERVAL> --live_seconds <LIVE_SECONDS> --max_inventory <MAX_INVENTORY> --max_notional <MAX_NOTIONAL> --auto_approve --dry_run --debug
```

Live — Liquidity Taker:
```bash
python main.py --token_id <TOKEN_ID> --mode live --interval <INTERVAL> --live_seconds <LIVE_SECONDS> --max_notional <MAX_NOTIONAL> --auto_approve --dry_run --debug
```

