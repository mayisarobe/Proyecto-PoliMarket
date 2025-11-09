# main.py
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import re
import time
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, parse_qs

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt

# --- Nuestras clases ---
from fairprice import FairPrice            # Kalman 1D mejorado (usa clip_prob/clip01)
from spread import SpreadCalculator        # Avellaneda–Stoikov mejorado

CLOB_BASE = "https://clob.polymarket.com"

# ============ Sesión HTTP robusta (evita 429/HTML/Cloudflare) ============
def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PolymarketMM/1.0",
        "Accept": "application/json",
        "Connection": "keep-alive",
    })
    retry = Retry(
        total=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

_SESSION: Optional[requests.Session] = None
def http_get_json(url: str, timeout: int = 20):
    """
    GET JSON estricto: si la respuesta no es application/json lanza error
    e incluye un snippet del body para debuggear HTML/Cloudflare.
    """
    global _SESSION
    if _SESSION is None:
        _SESSION = _make_session()
    r = _SESSION.get(url, timeout=timeout)
    ct = (r.headers.get("Content-Type") or "").lower()
    if "application/json" not in ct:
        snippet = r.text[:300].replace("\n", " ")
        raise RuntimeError(f"No-JSON {r.status_code} {ct} body[:300]={snippet!r} url={url}")
    return r.json()

# ============ Helpers de parsing ============
def _ensure_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return [s.strip() for s in x.split(",") if s.strip()]
    return []

def is_decimal_token_id(s: str) -> bool:
    # Los token_id del CLOB son enteros decimales muy largos
    return bool(re.fullmatch(r"\d{25,}", s))

def parse_polymarket_url(url: str) -> dict:
    """
    Soporta URLs tipo:
      - https://polymarket.com/event/<slug>
      - https://polymarket.com/market/<slug>?tid=<token_id_decimal>
      - https://polymarket.com/markets/<slug>
    """
    u = urlparse(url)
    out: Dict[str, str] = {}
    qs = parse_qs(u.query)
    if "tid" in qs:
        tid = qs["tid"][0]
        if is_decimal_token_id(tid):
            out["token_id"] = tid
    path = (u.path or "").strip("/")
    m = re.search(r"(?:event|markets?)/([^/?#]+)", path, flags=re.I)
    if m:
        out["slug"] = m.group(1)
    return out

# ============ Resolver token_id desde /markets ============
def _extract_markets_payload(obj) -> Optional[List[dict]]:
    """
    Devuelve lista de mercados desde distintas variantes de payload:
    - dict con 'markets' | 'data' | 'results'
    - lista de dicts directamente
    """
    if isinstance(obj, dict):
        for key in ("markets", "data", "results"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        # algunos backends devuelven {'markets': {'data': [...]}}
        if "markets" in obj and isinstance(obj["markets"], dict) and "data" in obj["markets"]:
            if isinstance(obj["markets"]["data"], list):
                return obj["markets"]["data"]
        return None
    if isinstance(obj, list):
        # asumimos lista de mercados
        return obj
    return None

def fetch_markets() -> List[dict]:
    """
    Intenta varias variantes de /markets. Devuelve lista de mercados o lanza error detallado.
    """
    candidates = [
        f"{CLOB_BASE}/markets?limit=1000",
        f"{CLOB_BASE}/markets",
        f"{CLOB_BASE}/markets?active=true",
    ]
    last_err = None
    for url in candidates:
        try:
            data = http_get_json(url)
            markets = _extract_markets_payload(data)
            if markets and isinstance(markets, list):
                return markets
        except Exception as e:
            last_err = e
            time.sleep(0.3)
            continue
    if last_err:
        raise RuntimeError(f"No se pudo obtener /markets de {candidates}. Último error: {last_err}")
    raise TypeError("Formato inesperado en /markets (sin 'markets'/'data'/'results' ni lista).")

def find_market_by_condition(condition_id_hex: str) -> dict:
    cid = condition_id_hex.lower()
    for m in fetch_markets():
        if str(m.get("conditionId", "")).lower() == cid:
            return m
    raise ValueError(f"conditionId no encontrado: {condition_id_hex}")

def find_market_by_slug(slug: str) -> dict:
    target = slug.strip("/").lower()
    for m in fetch_markets():
        mslug = (m.get("slug") or m.get("urlSlug") or "").strip("/").lower()
        if mslug == target:
            return m
    # fallback contiene
    for m in fetch_markets():
        mslug = (m.get("slug") or m.get("urlSlug") or "").strip("/").lower()
        if target in mslug:
            return m
    raise ValueError(f"Slug no encontrado: {slug}")

def outcome_to_token_map(market: dict) -> Tuple[str, Dict[str, str], List[str]]:
    question = market.get("question", "Unknown market")
    outcomes = _ensure_list(market.get("outcomes", []))
    clob_ids = _ensure_list(market.get("clobTokenIds", []))
    # fallback a estructura 'tokens'
    if (not outcomes or not clob_ids) and market.get("tokens"):
        outcomes = [t.get("outcome") for t in market["tokens"]]
        clob_ids = [t.get("id") for t in market["tokens"]]
    if len(outcomes) != len(clob_ids):
        raise ValueError("Descuadre outcomes vs clobTokenIds/tokens")
    mapping = {str(o): str(t) for o, t in zip(outcomes, clob_ids)}
    return question, mapping, outcomes

def resolve_token_from_market(market: dict, outcome: Optional[str], outcome_index: Optional[int]) -> str:
    _, mapping, outcomes = outcome_to_token_map(market)
    if outcome is not None:
        oq = outcome.lower()
        if outcome in mapping:
            return mapping[outcome]
        for o in outcomes:
            so = str(o)
            if oq == so.lower() or oq in so.lower():
                return mapping[o]
        raise SystemExit(f"Outcome '{outcome}' no coincide. Disponibles: {outcomes}")
    if outcome_index is not None:
        if not (0 <= outcome_index < len(outcomes)):
            raise SystemExit(f"Índice fuera de rango. 0..{len(outcomes)-1}")
        return mapping[outcomes[outcome_index]]
    raise SystemExit("Falta --outcome o --outcome_index para seleccionar token en mercado multi-opción.")

def resolve_token_id(args) -> str:
    if args.token_id:
        return args.token_id
    if args.url:
        parsed = parse_polymarket_url(args.url)
        if "token_id" in parsed:
            return parsed["token_id"]
        if "slug" not in parsed:
            raise SystemExit("URL sin 'tid' ni slug reconocible.")
        market = find_market_by_slug(parsed["slug"])
        return resolve_token_from_market(market, args.outcome, args.outcome_index)
    if args.condition_id:
        market = find_market_by_condition(args.condition_id)
        return resolve_token_from_market(market, args.outcome, args.outcome_index)
    raise SystemExit("No se pudo resolver token_id (revisa parámetros).")

# ============ Order book ============
def _first_price(arr):
    if not isinstance(arr, list) or not arr:
        return None
    top = arr[0]
    if isinstance(top, dict):
        p = top.get("price") or top.get("px")
        try:
            return float(p) if p is not None else None
        except Exception:
            return None
    if isinstance(top, (list, tuple)) and top:
        try:
            return float(top[0])
        except Exception:
            return None
    return None

def get_best_bid_ask(token_id: str, debug: bool = False):
    """
    Intenta varios endpoints del CLOB y devuelve (best_bid, best_ask) como floats o None.
    """
    for ep in ["orderbook", "book", "orders"]:
        url = f"{CLOB_BASE}/{ep}?token_id={token_id}"
        try:
            data = http_get_json(url)
            if isinstance(data, dict):
                bids = data.get("bids") or data.get("bid") or []
                asks = data.get("asks") or data.get("ask") or []
                bb = _first_price(bids)
                ba = _first_price(asks)
                if debug:
                    print(f"[debug] {ep}: bestBid={bb} bestAsk={ba}")
                if bb is not None or ba is not None:
                    return bb, ba
            elif isinstance(data, list):
                # /orders devuelve lista simple de órdenes
                bids = [o for o in data if str(o.get("side","")).lower()=="buy"]
                asks = [o for o in data if str(o.get("side","")).lower()=="sell"]
                bids.sort(key=lambda x: float(x.get("price") or x.get("px") or 0), reverse=True)
                asks.sort(key=lambda x: float(x.get("price") or x.get("px") or 0))
                bb = float(bids[0].get("price") or bids[0].get("px")) if bids else None
                ba = float(asks[0].get("price") or asks[0].get("px")) if asks else None
                if debug:
                    print(f"[debug] {ep}-list: bestBid={bb} bestAsk={ba}")
                return bb, ba
        except Exception as e:
            if debug:
                print(f"[debug] fallo {url}: {e}")
            time.sleep(0.5)
            continue
    return None, None

# ============ Aux mid ============
def safe_mid(bb, ba, last_mid: Optional[float], default_mid: float) -> float:
    if bb is None and ba is None:
        return last_mid if last_mid is not None else default_mid
    if bb is None:
        return max(0.01, min(0.99, ba - 0.01))
    if ba is None:
        return max(0.01, min(0.99, bb + 0.01))
    return (bb + ba) / 2.0

# ============ MAIN ============
def main():
    ap = argparse.ArgumentParser(
        description="Polymarket MM: FairPrice(Kalman) + Avellaneda–Stoikov + CSV/Gráfica."
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--token_id", type=str, help="Token ID decimal del CLOB (directo).")
    src.add_argument("--condition_id", type=str, help="ConditionId hex (0x...).")
    src.add_argument("--url", type=str, help="URL de Polymarket (event/<slug> o con ?tid=).")

    ap.add_argument("--outcome", type=str, help="Nombre/parcial del outcome (si condition/url).")
    ap.add_argument("--outcome_index", type=int, help="Índice del outcome (si condition/url).")
    ap.add_argument("--list_outcomes", action="store_true", help="Solo listar outcomes/token_id y salir.")

    # muestreo (más datos, mismo rango temporal)
    ap.add_argument("--samples", type=int, default=3600, help="Número de lecturas (p.ej., 3600=1h a 1s).")
    ap.add_argument("--interval", type=float, default=1.0, help="Segundos entre lecturas.")

    # FairPrice
    ap.add_argument("--initial_price", type=float, default=0.5)
    ap.add_argument("--process_variance", type=float, default=1e-5)      # Q base
    ap.add_argument("--measurement_variance", type=float, default=1e-2)  # R base
    ap.add_argument("--initial_variance", type=float, default=1.0)
    ap.add_argument("--clip01", action="store_true")

    # SpreadCalculator
    ap.add_argument("--gamma", type=float, default=0.15)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--min_spread", type=float, default=0.003)
    ap.add_argument("--inv", type=float, default=0.0)
    ap.add_argument("--inv_limit", type=float, default=30.0)
    ap.add_argument("--inv_skew_scale", type=float, default=1.0)

    # salida
    ap.add_argument("--png", type=str, default="mm_results.png")
    ap.add_argument("--csv", type=str, default="mm_timeseries.csv")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    # ===== Listado de outcomes (sin operar) =====
    if args.list_outcomes:
        if args.token_id:
            print("Con --token_id no hay outcomes que listar (ya es un único outcome).")
            return
        if args.url:
            parsed = parse_polymarket_url(args.url)
            if "slug" not in parsed:
                raise SystemExit("URL sin slug. Usa --url .../event/<slug> o --condition_id.")
            market = find_market_by_slug(parsed["slug"])
        elif args.condition_id:
            market = find_market_by_condition(args.condition_id)
        else:
            raise SystemExit("Para --list_outcomes usa --url <...> o --condition_id <0x...>")
        q, mapping, outcomes = outcome_to_token_map(market)
        print(f"== {q} ==")
        for i, o in enumerate(outcomes):
            print(f"[{i}] {o} -> {mapping[o]}")
        return

    # ===== Resolver token_id =====
    token_id = resolve_token_id(args)
    if args.debug:
        print(f"[debug] token_id={token_id}")

    # ===== Instancias (tus clases) =====
    # FairPrice: más memoria en Q dinámica (sin tocar fechas)
    kf = FairPrice(
        initial_price=args.initial_price,
        process_variance=args.process_variance,
        measurement_variance=args.measurement_variance,
        initial_variance=args.initial_variance,
        clip_prob=args.clip01,    # alias de clip01 aceptado por tu clase
        # tuning de micro-vol (si tu FairPrice los soporta)
        q_alpha=0.10,
        q_gain=0.15
    )

    # Spread: más memoria y sensibilidad micro
    sc = SpreadCalculator(
        gamma=args.gamma,
        lam=args.lam,
        window=200,          # horizonte lógico mayor
        ewma_alpha=0.15,     # sigma más memoriosa
        k_spread=0.35,
        k_imb=0.015,
        edge_k=0.30,
        hard_max=0.20
    )

    # ===== Buffers de serie =====
    times: List[float] = []
    mids: List[float] = []
    fairs: List[float] = []
    fair_vars: List[float] = []
    bidq: List[float] = []
    askq: List[float] = []
    best_bids: List[Optional[float]] = []
    best_asks: List[Optional[float]] = []

    t0 = time.time()
    last_mid: Optional[float] = None

    # ===== Loop principal =====
    for i in range(args.samples):
        bb, ba = get_best_bid_ask(token_id, debug=args.debug)
        mid = safe_mid(bb, ba, last_mid, args.initial_price)

        # Fair price
        if bb is not None and ba is not None:
            fair, var = kf.update_from_orderbook(bb, ba)
        else:
            fair, var = kf.update(mid)

        # Microestructura para spread
        if bb is not None and ba is not None:
            sc.update_micro(best_bid=bb, best_ask=ba)

        # Cotizaciones (Avellaneda–Stoikov)
        bid, ask, full = sc.quotes(fair=fair, fair_var=var, inv=args.inv, inv_limit=args.inv_limit)

        # almacenar
        now = time.time() - t0
        times.append(now)
        mids.append(mid)
        fairs.append(fair)
        fair_vars.append(var)
        bidq.append(bid)
        askq.append(ask)
        best_bids.append(bb if bb is not None else "")
        best_asks.append(ba if ba is not None else "")

        if args.debug:
            print(
                f"[{i+1}/{args.samples}] mid={mid:.4f} fair={fair:.4f} (P={var:.6f}) | "
                f"bid={bid:.4f} ask={ask:.4f} | bestBid={bb} bestAsk={ba}"
            )

        last_mid = mid
        if i < args.samples - 1:
            time.sleep(max(0.0, args.interval))

    # ===== CSV =====
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s","best_bid","best_ask","mid","fair","fair_var","quote_bid","quote_ask"])
        for row in zip(times, best_bids, best_asks, mids, fairs, fair_vars, bidq, askq):
            w.writerow(row)
    print(f"✅ Serie temporal guardada en: {args.csv}")

    # ===== Gráfica =====
    plt.figure(figsize=(10, 5))
    plt.plot(times, mids, label="Mid observado")
    plt.plot(times, fairs, label="Fair price (Kalman)")
    plt.plot(times, bidq, label="Bid cotizado")
    plt.plot(times, askq, label="Ask cotizado")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Precio")
    plt.title("Fair price (Kalman) + Avellaneda–Stoikov | Polymarket")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.png, dpi=150)
    print(f"📈 Gráfica guardada en: {args.png}")

    # ===== Resumen corto para el pitch =====
    avg_spread = sum(a - b for a, b in zip(askq, bidq)) / max(1, len(bidq))
    drift = (fairs[-1] - fairs[0]) if fairs else 0.0
    print(f"Resumen → spread_medio={avg_spread:.4f} | drift_fair={drift:+.4f}")

if __name__ == "__main__":
    main()



