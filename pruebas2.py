import sys, time, requests
from fairprice import FairPrice
from spread import SpreadCalculator

print("[probe] Python exe:", sys.executable, flush=True)

TOKEN_ID = "76184513907290761912636659055080401703643418316153242056949287928791438454394"
print("[probe] Fetching orderbook…", flush=True)
ob = requests.get(f"https://clob.polymarket.com/orderbook?token_id={TOKEN_ID}", timeout=10).json()
bids = ob.get("bids", [])
asks = ob.get("asks", [])
bb = float(bids[0].get("price", bids[0].get("px"))) if bids else None
ba = float(asks[0].get("price", asks[0].get("px"))) if asks else None
print(f"[probe] bestBid={bb} bestAsk={ba}", flush=True)

def safe_mid(bb, ba, last_mid=None, default=0.5):
    if bb is None and ba is None:
        return last_mid if last_mid is not None else default
    if bb is None: return max(0.01, min(0.99, ba - 0.01))
    if ba is None: return max(0.01, min(0.99, bb + 0.01))
    return (bb + ba)/2

kf = FairPrice(initial_price=0.5, process_variance=1e-5, measurement_variance=1e-2, clip01=True)
sc = SpreadCalculator(gamma=0.15, lam=0.6, window=50, min_spread=0.003)

mid = safe_mid(bb, ba, default=0.5)
print(f"[probe] mid0={mid:.4f}", flush=True)

for i in range(1,6):
    # reconsulta libro para ver que hay movimiento
    ob = requests.get(f"https://clob.polymarket.com/orderbook?token_id={TOKEN_ID}", timeout=10).json()
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    bb = float(bids[0].get("price", bids[0].get("px"))) if bids else None
    ba = float(asks[0].get("price", asks[0].get("px"))) if asks else None
    mid = safe_mid(bb, ba, last_mid=mid, default=0.5)

    if bb is not None and ba is not None:
        fair, var = kf.update_from_orderbook(bb, ba)
    else:
        fair, var = kf.update(mid)

    bid, ask, full = sc.quotes(fair=fair, fair_var=var, inv=0.0, inv_limit=30.0)
    print(f"[{i}/5] mid={mid:.4f} fair={fair:.4f} (P={var:.6f}) | quote_bid={bid:.4f} quote_ask={ask:.4f} | bestBid={bb} bestAsk={ba}", flush=True)
    time.sleep(0.5)

print("[probe] OK. Si ves estas líneas, la salida funciona.", flush=True)
