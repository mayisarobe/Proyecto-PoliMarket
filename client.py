from __future__ import annotations
import os
from typing import Optional, Literal
from dotenv import load_dotenv

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

LoginMode = Literal["magic", "wallet", "eoa"]  # email=magic, metamask=wallet, eoa=sin proxy

class PolymarketClient:
    def __init__(
        self,
        private_key: Optional[str] = None,
        login_mode: LoginMode = "wallet",
        proxy_address: Optional[str] = None,
        host: str = "https://clob.polymarket.com",
        chain_id: int = 137,
        auto_auth: bool = True,
    ) -> None:
        load_dotenv()
        key  = private_key or os.getenv("PRIVATE_KEY")
        if not key:
            raise ValueError("Falta PRIVATE_KEY (.env o parámetro).")

        proxy = proxy_address or os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        if login_mode == "magic":
            if not proxy: raise ValueError("Falta POLYMARKET_PROXY_ADDRESS para login_mode=magic.")
            self.c = ClobClient(host, key=key, chain_id=chain_id, signature_type=1, funder=proxy)
        elif login_mode == "wallet":
            if not proxy: raise ValueError("Falta POLYMARKET_PROXY_ADDRESS para login_mode=wallet.")
            self.c = ClobClient(host, key=key, chain_id=chain_id, signature_type=2, funder=proxy)
        elif login_mode == "eoa":
            self.c = ClobClient(host, key=key, chain_id=chain_id)
        else:
            raise ValueError("login_mode inválido: usa magic | wallet | eoa")

        if auto_auth:
            creds = self.c.create_or_derive_api_creds()
            self.c.set_api_creds(creds)

    