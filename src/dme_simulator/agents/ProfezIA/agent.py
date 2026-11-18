"""ProfezIA agent entry point (Owner: Persona 3).

Wires together the three independent strategy modules:
- `bidding.py`    (Persona 1)
- `production.py` (Persona 3)
- `pricing.py`    (Persona 2)
"""

from __future__ import annotations

from dme_simulator.agents import BaseAgent, Decision, Observation
from dme_simulator.config import GoodsConfig

from .bidding import BiddingStrategy
from .pricing import PricingStrategy
from .production import ProductionStrategy
from .state import MarketContext


class ProfezIAAgent(BaseAgent):
    """Modular team agent delegating each turn phase to its dedicated module."""

    def __init__(self) -> None:
        self.ctx = MarketContext()
        self.bidding = BiddingStrategy()
        self.production = ProductionStrategy()
        self.pricing = PricingStrategy()

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        self.ctx.update(state, goods_config)

        bids = self.bidding.decide_bids(state, goods_config, self.ctx)
        self.ctx.planned_bids = bids

        assembly = self.production.decide_assembly(state, goods_config, self.ctx)
        self.ctx.planned_assembly = assembly

        listings = self.pricing.decide_listings(state, goods_config, self.ctx)

        return Decision(bids=bids, assembly=assembly, listings=listings)
