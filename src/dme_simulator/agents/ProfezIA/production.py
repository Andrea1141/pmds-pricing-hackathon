"""Phase 2: Priority-ranked, capacity-aware production strategy for ProfezIA.

Assembles finished goods in descending margin-per-input priority order using
materials already held plus same-turn auction winnings, then performs a bounded
secondary pass to convert spare raw stock into active profitable goods.
"""

from __future__ import annotations

from dme_simulator.agents import AssemblyOrder, Observation
from dme_simulator.config import GoodsConfig

from .state import MarketContext


class ProductionStrategy:
    """Computes ordered `AssemblyOrder` instructions from unified `MarketContext`."""

    def __init__(self) -> None:
        pass

    def decide_assembly(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> list[AssemblyOrder]:
        """Return ordered `AssemblyOrder` list prioritizing highest-margin goods."""
        if not ctx.ranked_goods:
            return []

        orders: list[AssemblyOrder] = []
        # Pass 1: Planned build targets in descending priority order.
        for g in ctx.ranked_goods:
            qty = ctx.need_build.get(g, 0)
            rname = ctx.recipe_for.get(g)
            if qty > 0 and rname is not None:
                orders.append(AssemblyOrder(rname, int(qty)))

        # Pass 2: Bounded leftover assembly for profitable, actively-selling goods
        # so unused raw materials from previous partial fills convert into sellable stock.
        for g in ctx.ranked_goods:
            if ctx.zero_streak.get(g, 0) >= 2:
                continue
            rname = ctx.recipe_for.get(g)
            if rname is None:
                continue
            curr_inv = max(0, int(state.inventory.get(g, 0)))
            already_planned = ctx.need_build.get(g, 0)
            max_extra = max(
                0,
                min(6, ctx.target_stock.get(g, 2) + 2 - curr_inv - already_planned),
            )
            if max_extra > 0:
                orders.append(AssemblyOrder(rname, int(max_extra)))

        return orders
