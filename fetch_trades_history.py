# record_live_book.py
# -*- coding: utf-8 -*-
import argparse, csv, time, sys
from datetime import datetime, timezone
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://clob.polymarket.com"

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PolymarketMM/1.0",
        "Accept": "application/json",
        "Connection": "keep-alive",
    })
    retry = Retry(total=5, backoff_factor=0.8,
                  status_forcelist=[429,500,502,503,504],
                  respect_retry_after_header=True, allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def get_json(s, url):
    r = s.get(url, timeout=15)
    r.raise_for_status()
    if "application/json" not in r.headers.get("Content-Type","").lower():
        raise RuntimeError(f"No-JSON {r.status_code} {r.headers.get('Content-Type')} @ {url}")
    return r.json()

def first_px(arr):
    if not isinstance(arr, list) or not arr:
        return None
    top = arr[0]
    if isinstance(top, dict):
        p = top.get("price") or top.get("px")
        try: return float(p) if p is not None else None
        except: return None
    if isinstance(top, (list, tuple)) and top:
        try: return float(top[0])
        except: return None
    return None

def main():
    ap = argparse.ArgumentParser("Graba el libro en vivo (/book) a CSV durante un tiempo dado.")
    ap.add_argument("--token_id", required=True)
    ap.add_argument("--duration_s", type=int, default=1800, help="Duración en segundos (default 30 min).")
    ap.add_argument("--interval", type=float, default=1.0, help="Segundos entre lecturas.")
    ap.add_argument("--out", type=str, default="live_book.csv")
    args = ap.parse_args()

    s = make_session()
    start = time.time()
    rows = 0

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_iso","best_bid","best_ask"])
        while True:
            now = time.time()
            if now - start >= args.duration_s:
                break
            try:
                data = get_json(s, f"{BASE}/book?token_id={args.token_id}")
                bb = first_px(data.get("bids") or data.get("bid") or [])
                ba = first_px(data.get("asks") or data.get("ask") or [])
                ts = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
                w.writerow([ts, "" if bb is None else bb, "" if ba is None else ba])
                rows += 1
                # feedback ligero cada 60 puntos
                if rows % 60 == 0:
                    print(f"[rec] {rows} muestras…", flush=True)
            except Exception as e:
                print(f"[rec] fallo lectura: {e}", file=sys.stderr)
            time.sleep(max(0.0, args.interval))
    print(f"✅ Grabación terminada: {rows} filas en {args.out}")

if __name__ == "__main__":
    main()
