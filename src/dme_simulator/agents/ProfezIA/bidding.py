"""Phase 1: Raw-material auction bidding strategy (Owner: Persona 1).

Keep all bidding logic and any bidding-specific internal state inside this
module (e.g. attributes on `self`) so changes never conflict with other files.
"""

from __future__ import annotations

from dme_simulator.agents import Bid, Observation
from dme_simulator.config import GoodsConfig

from .state import MarketContext


class BiddingStrategy:
    """Computes sealed pay-your-bid offers for each turn's raw-material lots."""

    def __init__(self) -> None:
        # Store bidding-specific state here (e.g., clearing-price EMAs, win rates).
        pass

    def decide_bids(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> list[Bid]:
        """Return sealed bids for this turn's `state.auction_lots`."""
        bids: list[Bid] = []
        for lot in state.auction_lots:
            if lot.quantity <= 0:
                continue
            bids.append(Bid(lot.good, 1, lot.reserve_price * 1.1))
        return bids
