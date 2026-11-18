"""Phase 3: 1-unit softmax price-ladder strategy for ProfezIA.

Lists each planned finished unit as its own 1-unit `Listing` on a price ladder
around the good's reference price:
- Maximizes legitimate softmax shelf share in `_clear_sales`.
- Performs built-in 2nd-degree price discrimination across customer segments.
- Uses upper-rung and sellout outcomes as clean feedback for `ctx.ref_price`.
- Includes an early probe rung when a good has not sold yet to find low-WTP
  markets rapidly.
"""

from __future__ import annotations

from math import ceil, isfinite

from dme_simulator.agents import Listing, Observation
from dme_simulator.config import GoodsConfig

from .state import MarketContext


class PricingStrategy:
    """1-unit price-ladder listing strategy with early discovery probes."""

    ladder_low: float = 0.60
    monopoly_low: float = 0.72
    ladder_high: float = 1.00
    early_low: float = 0.75
    early_high: float = 1.60
    probe_mult: float = 0.55
    max_listings_per_good: int = 160

    def __init__(self) -> None:
        pass

    @staticmethod
    def _round_price(value: float) -> float:
        decimals = 4 if value < 5.0 else 2
        return max(0.01, round(float(value), decimals))

    def decide_listings(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> list[Listing]:
        """Return 1-unit ladder `Listing` offers for `goods_config.finished_goods`."""
        early = state.turn <= ctx.explore_turns
        inv = {k: int(v) for k, v in state.inventory.items()}

        listings: list[Listing] = []
        last_ladder: dict[str, list[float]] = {}
        my_prices: set[float] = set()

        for g in goods_config.finished_goods:
            ref = ctx.ref_price.get(g, 0.0)
            if not isfinite(ref) or ref <= 0.0:
                continue

            planned = max(0, inv.get(g, 0)) + max(0, ctx.need_build.get(g, 0))
            if planned <= 0:
                continue

            floor = ctx.active_floor(g)
            if early:
                low, high = self.early_low, self.early_high
            else:
                low = (
                    self.monopoly_low
                    if ctx.rival_stock.get(g, 0) == 0
                    else self.ladder_low
                )
                high = self.ladder_high

            n = planned
            prices: list[float] = []
            for i in range(n):
                frac = 0.5 if n == 1 else i / float(n - 1)
                p = ref * (low + (high - low) * frac)
                prices.append(self._round_price(max(p, floor, 0.01)))

            # Early discovery probe: if a good has never sold by turn >= 2, replace the
            # lowest rung with a cheap probe so low-WTP markets are discovered immediately.
            if state.turn >= 2 and ctx.total_sold.get(g, 0) == 0 and n >= 3:
                probe_floor = max(0.01, ctx.good_reserve_cost.get(g, 0.0) * 1.08)
                probe_p = self._round_price(max(probe_floor, ref * self.probe_mult))
                if probe_p < prices[0]:
                    prices[0] = probe_p

            ladder = sorted(prices)
            last_ladder[g] = ladder

            if n <= self.max_listings_per_good:
                for p in ladder:
                    if isfinite(p) and p > 0.0:
                        listings.append(Listing(g, 1, float(p)))
                        my_prices.add(round(float(p), 4))
            else:
                chunk = ceil(n / float(self.max_listings_per_good))
                for start in range(0, n, chunk):
                    sub = ladder[start : start + chunk]
                    avg_p = self._round_price(sum(sub) / len(sub))
                    if isfinite(avg_p) and avg_p > 0.0:
                        listings.append(Listing(g, len(sub), float(avg_p)))
                        my_prices.add(round(float(avg_p), 4))

            # Catchall buffer in case leftover assembly builds a few bonus units.
            catchall_p = self._round_price(max(ref, floor, 0.01))
            listings.append(Listing(g, 6, float(catchall_p)))
            my_prices.add(round(float(catchall_p), 4))

        ctx.last_ladder = last_ladder
        ctx.my_list_prices = my_prices
        return listings
