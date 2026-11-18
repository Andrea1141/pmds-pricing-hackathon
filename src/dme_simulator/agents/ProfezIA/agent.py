"""ProfezIA agent entry point (Owner: Persona 3).

Wires together the three independent strategy modules with fault isolation and
strict decision sanitization so a bug in one module never invalidates the turn:
- `bidding.py`    (Persona 1)
- `production.py` (Persona 3)
- `pricing.py`    (Persona 2)
"""

from __future__ import annotations

import logging
from math import isfinite
from numbers import Integral, Real
from typing import Callable, TypeVar

from dme_simulator.agents import (
    AssemblyOrder,
    BaseAgent,
    Bid,
    Decision,
    Listing,
    Observation,
)
from dme_simulator.config import GoodsConfig

from .bidding import BiddingStrategy
from .pricing import PricingStrategy
from .production import ProductionStrategy
from .state import MarketContext

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class ProfezIAAgent(BaseAgent):
    """Modular team agent delegating each turn phase to its dedicated module."""

    def __init__(self) -> None:
        self.ctx = MarketContext()
        self.bidding = BiddingStrategy()
        self.production = ProductionStrategy()
        self.pricing = PricingStrategy()

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        try:
            self.ctx.update(state, goods_config)
        except Exception:
            logger.warning(
                "ProfezIA: MarketContext.update failed on turn %d",
                state.turn,
                exc_info=True,
            )
            self.ctx.turn = state.turn
            self.ctx.new_auction_txs = []
            self.ctx.new_sales_txs = []
            self.ctx.planned_bids = []
            self.ctx.planned_assembly = []

        raw_bids = self._safe_call(
            "bidding", self.bidding.decide_bids, state, goods_config, self.ctx
        )
        bids = self._sanitize_bids(raw_bids, state)
        self.ctx.planned_bids = bids

        raw_assembly = self._safe_call(
            "production", self.production.decide_assembly, state, goods_config, self.ctx
        )
        assembly = self._sanitize_assembly(raw_assembly, goods_config)
        self.ctx.planned_assembly = assembly

        raw_listings = self._safe_call(
            "pricing", self.pricing.decide_listings, state, goods_config, self.ctx
        )
        listings = self._sanitize_listings(raw_listings, goods_config)

        return Decision(bids=bids, assembly=assembly, listings=listings)

    @staticmethod
    def _safe_call(
        phase: str,
        fn: Callable[[Observation, GoodsConfig, MarketContext], list[_T]],
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> list[_T]:
        """Run one phase isolated from normal exceptions (never catches DecisionTimeout)."""
        try:
            result = fn(state, goods_config, ctx)
            return result if isinstance(result, list) else list(result)
        except Exception:
            logger.warning(
                "ProfezIA: %s module raised an exception on turn %d; falling back to []",
                phase,
                state.turn,
                exc_info=True,
            )
            return []

    @staticmethod
    def _to_positive_int(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, Integral):
            val = int(value)
            return val if val > 0 else None
        if isinstance(value, Real) and isfinite(float(value)):
            val = int(value)
            return val if val > 0 else None
        return None

    @staticmethod
    def _to_finite_nonneg_float(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        val = float(value)
        return val if isfinite(val) and val >= 0.0 else None

    @classmethod
    def _sanitize_bids(cls, raw_bids: list[Bid], state: Observation) -> list[Bid]:
        auction_goods = {lot.good for lot in state.auction_lots}
        clean: list[Bid] = []
        for item in raw_bids:
            if not isinstance(item, Bid) or item.good not in auction_goods:
                continue
            qty = cls._to_positive_int(item.quantity)
            price = cls._to_finite_nonneg_float(item.price_per_unit)
            if qty is not None and price is not None:
                clean.append(Bid(item.good, qty, price))
        return clean

    @classmethod
    def _sanitize_assembly(
        cls, raw_assembly: list[AssemblyOrder], goods_config: GoodsConfig
    ) -> list[AssemblyOrder]:
        recipes = goods_config.recipes
        clean: list[AssemblyOrder] = []
        for item in raw_assembly:
            if not isinstance(item, AssemblyOrder) or item.recipe_name not in recipes:
                continue
            qty = cls._to_positive_int(item.quantity)
            if qty is not None:
                clean.append(AssemblyOrder(item.recipe_name, qty))
        return clean

    @classmethod
    def _sanitize_listings(
        cls, raw_listings: list[Listing], goods_config: GoodsConfig
    ) -> list[Listing]:
        finished = set(goods_config.finished_goods)
        clean: list[Listing] = []
        for item in raw_listings:
            if not isinstance(item, Listing) or item.good not in finished:
                continue
            qty = cls._to_positive_int(item.quantity)
            price = cls._to_finite_nonneg_float(item.price)
            if qty is not None and price is not None and price > 0.0:
                clean.append(Listing(item.good, qty, price))
        return clean
