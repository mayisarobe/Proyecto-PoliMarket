# 💻 Market Making Algorítmico en Polymarket - Pitch a un Venture Capital

## 🖊️ Descripción
Desarrollo e implementación de un sistema de market making algorítmico orientado a operar en Polymarket. 

Este proyecto forma parte de la práctica final del curso.  
El objetivo es desarrollar un **algoritmo de market making o liquidity taker**
en el mercado de predicciones **Polymarket**, tomando como caso de estudio
el mercado del **ganador de la UEFA Champions League**.

## 🧩 Estructura del Proyecto

- `client.py` → conexión con la API de Polymarket (ClobClient).
- `fair_price.py` → calcula el precio justo (v1: midprice).
- `strategy.py` → define la lógica de compra/venta.
- `main.py` → orquesta todo (con flag de simulación).
- `.env` → guarda las claves privadas (no se sube al repo).
- `.gitignore` → evita subir archivos sensibles o temporales.

## ⚙️ Instalación
```bash
pip install -r requirements.txt
python main.py

