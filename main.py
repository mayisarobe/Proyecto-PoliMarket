# main.py
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import re
import time
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, parse_qs
# ============================================================
import os
from datetime import datetime, timezone


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt

# --- Nuestras clases ---
from fairprice import FairPrice            # Kalman 1D mejorado (usa clip_prob/clip01)
from spread import SpreadCalculator        # Avellaneda–Stoikov mejorado

CLOB_BASE = "https://clob.polymarket.com"

# ============================================================
#                 VISUALIZACIÓN / GRÁFICAS
# ============================================================

class TickLogger:
    def __init__(self, path: str):
        self.path = path
        self._f = None
        self._w = None
        self._header_written = False

    def open(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=[
            "ts",
            "best_bid",
            "best_ask",
            "mid",
            "fair",
            "bid",
            "ask",
            "spread",
            "inventory"
        ])

        self._w.writeheader()
        self._header_written = True

    def log(self, *, ts, best_bid, best_ask, mid, fair, bid, ask, spread=None, inventory=None):
        if not self._header_written:
            self.open()

        def _f(x):
            return "" if x is None else float(x)

        self._w.writerow({
            "ts": ts,
            "best_bid": _f(best_bid),
            "best_ask": _f(best_ask),
            "mid": _f(mid),
            "fair": _f(fair),
            "bid": _f(bid),
            "ask": _f(ask),
            "spread": "" if spread is None else _f(spread),
            "inventory": "" if inventory is None else _f(inventory),

        })
        self._f.flush()

    def close(self):
        if self._f:
            self._f.close()
            self._f = None
            self._w = None
            self._header_written = False



# ============================================================
#          SESIÓN HTTP ROBUSTA (CONEXIÓN AL MERCADO)
# ============================================================
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

# ============================================================
#                 HELPERS DE PARSING / MERCADOS
# ============================================================
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

# ============================================================
#                      ORDER BOOK / MID
# ============================================================
def _best_price(levels, side: str):
    """
    levels: lista de niveles del libro (bids/asks) del CLOB.
    side: "bid" o "ask".

    Devuelve:
      - para bids: max(price)
      - para asks: min(price)
    """
    if not isinstance(levels, list) or not levels:
        return None

    prices = []
    for lvl in levels:
        if isinstance(lvl, dict):
            p = lvl.get("price") or lvl.get("px")
        elif isinstance(lvl, (list, tuple)) and lvl:
            p = lvl[0]
        else:
            p = None

        try:
            if p is not None:
                prices.append(float(p))
        except (TypeError, ValueError):
            continue

    if not prices:
        return None

    if side.lower() == "bid":
        return max(prices)
    else:  # "ask"
        return min(prices)


def get_best_bid_ask(token_id: str, debug: bool = False):
    """
    Devuelve (best_bid, best_ask) como floats o None.
    """
    for ep in ["book", "orders"]:
        url = f"{CLOB_BASE}/{ep}?token_id={token_id}"
        try:
            data = http_get_json(url)
            # --- Caso: /book (dict con bids/asks) ---
            if isinstance(data, dict):
                bids = data.get("bids") or data.get("bid") or []
                asks = data.get("asks") or data.get("ask") or []

                bb = _best_price(bids, "bid")
                ba = _best_price(asks, "ask")

                if debug:
                    print(f"[debug] {ep}: bestBid={bb} bestAsk={ba}")
                if bb is not None or ba is not None:
                    return bb, ba

            # --- Caso: /orders (lista de órdenes crudas) ---
            elif isinstance(data, list):
                bids = [o for o in data if str(o.get("side", "")).lower() == "buy"]
                asks = [o for o in data if str(o.get("side", "")).lower() == "sell"]

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



def safe_mid(bb, ba, last_mid: Optional[float], default_mid: float) -> float:
    if bb is None and ba is None:
        return last_mid if last_mid is not None else default_mid
    if bb is None:
        return max(0.01, min(0.99, ba - 0.01))
    if ba is None:
        return max(0.01, min(0.99, bb + 0.01))
    return (bb + ba) / 2.0

# ============================================================
#          CLIENTE DE TRADING (ABSTRACTO / ENCHUFABLE)
# ============================================================
class TradingClient:
    """
    Capa fina para mandar órdenes.
    Por defecto está en modo 'dry-run' y solo imprime.
    Aquí puedes enchufar tu client real de Polymarket (API key, firma, etc).

    NUEVO:
    - require_approval: si True, pide confirmación manual por consola antes de enviar.
    - send_* devuelve bool indicando si se ha enviado la orden.
    """
    def __init__(self, dry_run: bool = True, require_approval: bool = True):
        self.dry_run = dry_run
        self.require_approval = require_approval
        self.client = None  # Aquí enchufarías tu SDK real

    def _confirm(self, msg: str) -> bool:
        """
        Pide aprobación manual de la orden.
        Si require_approval=False, aprueba siempre.
        """
        if not self.require_approval:
            return True
        ans = input(msg + " ¿Aprobar? [y/N] ").strip().lower()
        return ans.startswith("y")

    def send_limit_order(self, token_id: str, side: str, price: float, size: float) -> bool:
        side = side.upper()
        msg = f"[ORDER][LIMIT] token={token_id} side={side} px={price:.4f} size={size:.2f}"

        if self.dry_run:
            print(msg, " (dry-run, no enviada)")
            return False

        if not self._confirm(msg):
            print("[ORDER] Rechazada por el aprobador.")
            return False

        # self.client.create_order(token_id, side, price, size, reduce_only=False)
        print(msg, " (ENVIADA REAL)")
        return True

    def send_market_taker(
        self,
        token_id: str,
        side: str,
        size: float,
        ref_price: Optional[float] = None
    ) -> bool:
        side = side.upper()
        extra = f" ref_px={ref_price:.4f}" if ref_price is not None else ""
        msg = f"[ORDER][MARKET-TAKER] token={token_id} side={side} size={size:.2f}{extra}"

        if self.dry_run:
            print(msg, " (dry-run, no enviada)")
            return False

        if not self._confirm(msg):
            print("[ORDER] Rechazada por el aprobador.")
            return False

        # self.client.create_order(token_id, side, price=None, size=size, type='market', ...)
        print(msg, " (ENVIADA REAL)")
        return True

    def cancel_open_orders(self, token_id: str):
        """
        Cancela todas las órdenes abiertas del token.
        En real: aquí llamas a la API de cancelación masiva.
        """
        msg = f"[ORDER][CANCEL-ALL] token={token_id}"
        if self.dry_run:
            print(msg, " (dry-run, no cancelado en el exchange)")
            return
        # if not self._confirm(msg):
        #     print("[ORDER] Cancelación rechazada por el aprobador.")
        #     return
        # self.client.cancel_all(token_id=token_id)
        print(msg, " (CANCEL-ALL ENVIADA REAL)")

# ============================================================
#                           MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Polymarket MM: FairPrice(Kalman) + Avellaneda–Stoikov + CSV/Gráfica + modo live."
    )

    # --- selección de mercado ---
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--token_id", type=str, help="Token ID decimal del CLOB (directo).")
    src.add_argument("--condition_id", type=str, help="ConditionId hex (0x...).")
    src.add_argument("--url", type=str, help="URL de Polymarket (event/<slug> o con ?tid=).")

    ap.add_argument("--outcome", type=str, help="Nombre/parcial del outcome (si condition/url).")
    ap.add_argument("--outcome_index", type=int, help="Índice del outcome (si condition/url).")
    ap.add_argument("--list_outcomes", action="store_true", help="Solo listar outcomes/token_id y salir.")

    # --- modo de operación ---
    ap.add_argument("--mode", choices=["backtest", "live"], default="backtest",
                    help="backtest = solo serie + CSV + plot; live = lógica de trading en tiempo real.")
    ap.add_argument("--role", choices=["maker", "taker"], default="maker",
                    help="maker = cotiza bid/ask; taker = ejecuta solo con edge.")
    ap.add_argument("--order_size", type=float, default=10.0,
                    help="Tamaño de la orden (en unidades del contrato).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Si está activo, NO envía órdenes reales, solo imprime.")
    ap.add_argument("--taker_edge", type=float, default=0.01,
                    help="Edge mínimo (en probabilidad) para que el taker actúe (fair vs best bid/ask).")
    ap.add_argument("--auto_approve", action="store_true",
                    help="Si se activa, no pide confirmación manual para ejecutar órdenes.")

    # --- muestreo (para backtest y para live logging) ---
    ap.add_argument("--samples", type=int, default=3600,
                    help="Número de lecturas en backtest (p.ej., 3600=1h a 1s).")
    ap.add_argument("--interval", type=float, default=1.0, help="Segundos entre lecturas.")
    ap.add_argument("--live_seconds", type=int, default=0,
                    help="Duración en modo live (0 = infinito hasta Ctrl+C).")

    # --- FairPrice ---
    ap.add_argument("--initial_price", type=float, default=0.5)
    ap.add_argument("--process_variance", type=float, default=1e-5)      # Q base
    ap.add_argument("--measurement_variance", type=float, default=1e-2)  # R base
    ap.add_argument("--initial_variance", type=float, default=1.0)
    ap.add_argument("--clip01", action="store_true")

    # --- SpreadCalculator ---
    ap.add_argument("--gamma", type=float, default=0.15)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--min_spread", type=float, default=0.003)
    ap.add_argument("--inv", type=float, default=0.0)
    ap.add_argument("--inv_limit", type=float, default=30.0)
    ap.add_argument("--inv_skew_scale", type=float, default=1.0)

    # --- límites de riesgo / inventario (NUEVO) ---
    ap.add_argument("--max_inventory", type=float, default=100.0,
                    help="Límite absoluto de inventario (en contratos).")
    ap.add_argument("--max_notional", type=float, default=100.0,
                    help="Límite aproximado de exposición nocional (|pos|*precio).")

    # --- salida backtest ---
    ap.add_argument("--png", type=str, default="mm_results.png")
    ap.add_argument("--csv", type=str, default="mm_timeseries.csv")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    logger = TickLogger("replay.csv")
    logger.open()

    try:


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
            print(f"[debug] mode={args.mode}, role={args.role}, dry_run={args.dry_run}")

        # ===== Instancias (tus clases) =====
        kf = FairPrice(
            initial_price=args.initial_price,
            process_variance=args.process_variance,
            measurement_variance=args.measurement_variance,
            initial_variance=args.initial_variance,
            clip_prob=args.clip01,
            q_alpha=0.10,
            q_gain=0.15
        )

        sc = SpreadCalculator(
            gamma=args.gamma,
            lam=args.lam,
            window=200,
            ewma_alpha=0.15,
            k_spread=0.35,
            k_imb=0.015,
            edge_k=0.30,
            hard_max=0.20
        )

        client = TradingClient(
            dry_run=args.dry_run,
            require_approval=not args.auto_approve
        )

        # ===== Buffers de serie (backtest y logging) =====
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

        # ===== Estado de inventario / cash (NUEVO) =====
        # Empiezas en args.inv por si quieres partir de una posición no nula
        inventory: float = args.inv
        cash: float = 0.0  # PnL acumulado en prob units (aprox f*pos)

        def check_limits(side: str, size: float, price: float) -> bool:
            """
            Comprueba límites de inventario y exposición antes de mandar una orden.
            """
            nonlocal inventory
            s = side.upper()
            delta = size if s == "BUY" else -size
            new_inv = inventory + delta
            if abs(new_inv) > args.max_inventory:
                if args.debug:
                    print(f"[RISK] Límite de inventario superado: {new_inv} > {args.max_inventory}")
                return False

            notional_after = abs(new_inv * price)
            if notional_after > args.max_notional:
                if args.debug:
                    print(f"[RISK] Límite nocional superado: {notional_after:.4f} > {args.max_notional:.4f}")
                return False

            return True

        # ============================================================
        #                 LÓGICA DE UN PASO DE MERCADO
        # ============================================================
        def one_step(now: float):
            nonlocal last_mid, inventory, cash

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
            # IMPORTANTE: pasamos el INVENTARIO actual para que ajuste el precio de reserva
            bid, ask, full = sc.quotes(
                fair=fair,
                fair_var=var,
                inv=inventory,
                inv_limit=args.inv_limit
            )

            # ===== CSV replay =====
            ts = datetime.now(timezone.utc).isoformat()
            logger.log(
                ts=ts,
                best_bid=bb,
                best_ask=ba,
                mid=mid,
                fair=fair,
                bid=bid,
                ask=ask,
                spread=(ask - bid) if (ask is not None and bid is not None) else None,
                inventory=inventory
            )


            # ---- Registro de serie (para backtest / logging) ----
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
                    f"[step] t={now:.1f}s mid={mid:.4f} fair={fair:.4f} (P={var:.6f}) | "
                    f"bid={bid:.4f} ask={ask:.4f} | bestBid={bb} bestAsk={ba} | "
                    f"inv={inventory:.2f} cash={cash:.2f}"
                )

            # ---- Estrategia de trading según rol ----
            if args.mode == "live":
                if args.role == "maker":
                    if bb is None or ba is None:
                        if args.debug:
                            print("[MAKER] Libro vacío, no coto.")
                        last_mid = mid
                        return

                    # Asegurar quotes dentro de [0.01, 0.99]
                    q_bid = max(0.01, min(0.99, bid))
                    q_ask = max(0.01, min(0.99, ask))

                    # Evitar quotes cruzadas
                    if q_bid >= q_ask:
                        mid_raw = (bb + ba) / 2.0
                        half_spread_min = 0.003
                        q_bid = max(0.01, mid_raw - half_spread_min)
                        q_ask = min(0.99, mid_raw + half_spread_min)

                    # Cancelamos órdenes anteriores antes de cotizar las nuevas
                    client.cancel_open_orders(token_id)

                    # Chequeo de límites antes de poner las órdenes
                    if args.debug:
                        print(f"[MAKER] quoting bid={q_bid:.4f}, ask={q_ask:.4f}, size={args.order_size}")

                    # Para maker, el inventario real debería actualizarse con fills reales
                    # (por ahora asumimos que se gestiona fuera, aquí solo cotizamos)
                    if check_limits("BUY", args.order_size, q_bid):
                        client.send_limit_order(token_id, "BUY", q_bid, args.order_size)
                    else:
                        if args.debug:
                            print("[MAKER] BID bloqueado por límites de riesgo.")

                    if check_limits("SELL", args.order_size, q_ask):
                        client.send_limit_order(token_id, "SELL", q_ask, args.order_size)
                    else:
                        if args.debug:
                            print("[MAKER] ASK bloqueado por límites de riesgo.")

                elif args.role == "taker":
                    # Liquidity Taker: ejecuta solo cuando hay edge suficiente
                    if bb is None or ba is None:
                        last_mid = mid
                        return

                    # Edge: fair vs niveles del book
                    edge_buy = fair - ba      # si fair > ask ⇒ queremos comprar
                    edge_sell = bb - fair     # si bid > fair ⇒ queremos vender

                    acted = False

                    # BUY (tomar el ask)
                    if edge_buy > args.taker_edge:
                        price = ba
                        if check_limits("BUY", args.order_size, price):
                            print(f"[TAKER] BUY signal → fair={fair:.4f} > ask={ba:.4f} (edge={edge_buy:.4f})")
                            sent = client.send_market_taker(
                                token_id, "BUY", args.order_size, ref_price=price
                            )
                            if sent:
                                delta = args.order_size
                                inventory += delta
                                cash -= delta * price
                                acted = True
                        elif args.debug:
                            print(f"[TAKER] BUY bloqueado por límites de riesgo (edge={edge_buy:.4f}).")

                    # SELL (pegar al bid)
                    if edge_sell > args.taker_edge:
                        price = bb
                        if check_limits("SELL", args.order_size, price):
                            print(f"[TAKER] SELL signal → bid={bb:.4f} > fair={fair:.4f} (edge={edge_sell:.4f})")
                            sent = client.send_market_taker(
                                token_id, "SELL", args.order_size, ref_price=price
                            )
                            if sent:
                                delta = -args.order_size
                                inventory += delta
                                cash -= delta * price
                                acted = True
                        elif args.debug:
                            print(f"[TAKER] SELL bloqueado por límites de riesgo (edge={edge_sell:.4f}).")

                    if not acted and args.debug:
                        print(f"[TAKER] No action, edges: buy={edge_buy:.4f}, sell={edge_sell:.4f}")

            last_mid = mid
            if args.debug:
                print(f"[mid] bb={bb} ba={ba} mid={mid}")

        # ============================================================
        #                     EJECUCIÓN SEGÚN MODO
        # ============================================================
        if args.mode == "backtest":
            # ===== Loop principal de backtest =====
            for i in range(args.samples):
                now = time.time() - t0
                one_step(now)
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

        else:
            # ===== MODO LIVE: estrategia de trading en tiempo real =====
            print(f"🚀 Modo LIVE iniciado | role={args.role} | dry_run={args.dry_run} | order_size={args.order_size}")
            start = time.time()
            step_idx = 0
            while True:
                now = time.time() - t0
                one_step(now)
                step_idx += 1
                time.sleep(max(0.0, args.interval))

                if args.live_seconds > 0 and (time.time() - start) >= args.live_seconds:
                    print("⏱️ live_seconds alcanzado, saliendo de modo live.")
                    break
    finally:
        logger.close()
        # En live normalmente no guardas gráfico, pero tienes las series en memoria si quieres.

if __name__ == "__main__":
    main()
