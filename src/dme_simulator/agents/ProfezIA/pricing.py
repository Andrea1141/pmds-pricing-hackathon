"""Phase 3: Finished-goods pricing & sales strategy (Owner: Persona 2).

Keep all pricing logic and any pricing-specific internal state inside this
module (e.g. attributes on `self`) so changes never conflict with other files.
"""

from __future__ import annotations

from dme_simulator.agents import Listing, Observation
from dme_simulator.config import GoodsConfig

from .state import MarketContext


class PricingStrategy:
    """Computes sales listings (quantities and prices) for finished goods."""

    def __init__(self) -> None:
        # Store pricing-specific state here (e.g., demand curves, price bandits).
        pass

    def decide_listings(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> list[Listing]:
        """Return `Listing` offers for `goods_config.finished_goods`."""
        return [
            Listing(good, 1_000, 250.0)
            for good in goods_config.finished_goods
        ]
