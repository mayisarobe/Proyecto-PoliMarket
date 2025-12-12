import requests
from urllib.parse import urlparse
import json  # <- IMPORTANTE

GAMMA = "https://gamma-api.polymarket.com"

def _slug_from_url(url: str) -> str:
    path = urlparse(url).path
    return path.rstrip("/").split("/")[-1]

def _get_json(url, **params):
    r = requests.get(url, params=params or None, timeout=20)
    r.raise_for_status()
    return r.json()

def _to_list(x):
    """
    Convierte lo que venga (str con JSON, lista, None...) en una lista de Python.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        # Si parece JSON (empieza por [), intentamos json.loads
        if x.startswith("["):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                pass
        # Fallback: split por comas
        return [s.strip().strip('"') for s in x.strip("[]").split(",") if s.strip()]
    # Último recurso: iterarlo
    return list(x)

def _zip_outcomes_ids(outcomes_raw, clob_ids_raw):
    outcomes = _to_list(outcomes_raw)
    ids = _to_list(clob_ids_raw)

    n = min(len(outcomes), len(ids))
    pairs = []
    for i in range(n):
        pairs.append({"outcome": outcomes[i], "token_id": ids[i]})
    for j in range(n, len(ids)):
        pairs.append({"outcome": None, "token_id": ids[j]})
    return pairs

def tokens_from_polymarket_url(url: str):
    """
    Si es MARKET: devuelve [{'outcome', 'token_id'}, ...].
    Si es EVENT: devuelve [{'market_slug', 'question', 'pairs': [...]}, ...].
    Usa solo Gamma.
    """
    slug = _slug_from_url(url)

    # 1) Intentar como MARKET por slug
    try:
        m = _get_json(f"{GAMMA}/markets/slug/{slug}")
        pairs = _zip_outcomes_ids(m.get("outcomes"), m.get("clobTokenIds"))
        return pairs
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 404:
            raise

    # 2) Intentar como EVENT por slug
    evs = _get_json(f"{GAMMA}/events", slug=slug, limit=1)
    if isinstance(evs, list) and evs:
        ev = evs[0]
        results = []
        for mk in ev.get("markets", []):
            pairs = _zip_outcomes_ids(mk.get("outcomes"), mk.get("clobTokenIds"))
            results.append({
                "market_slug": mk.get("slug"),
                "question": mk.get("question"),
                "pairs": pairs
            })
        return results

    # 3) Nada encontrado
    return []

if __name__ == "__main__":
    res = tokens_from_polymarket_url("https://polymarket.com/market/first-to-5k-gold-or-eth")
    for p in res:
        print(p)
