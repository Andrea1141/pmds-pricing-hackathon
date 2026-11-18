"""Phase 1: Adaptive pay-your-bid auction strategy for ProfezIA.

Combines:
- Capital-protected probing in hyper-crowded tiny-supply rooms (limits downside
  to <=2.5% of starting capital until demand/profit is proven, then unleashes
  full scaling in wave/extended markets)
- Starving-gated multiplicative escalation & fast full-fill decay (pay-your-bid)
- Lot-growth detection (`lot_aware`) to shed premium rapidly in surplus
- Adaptive bid-cap multiplier expanding smoothly up to 60x in high-demand waves
- Per-good kit-profitability gate (never lets cheap goods cap bids for luxury goods)
- Kit-aligned budget allocation in descending recipe-priority order
"""

from __future__ import annotations

from math import isfinite

from dme_simulator.agents import Bid, Observation
from dme_simulator.config import GoodsConfig

from .state import MarketContext


class BiddingStrategy:
    """Adaptive sealed-bid procurement optimized for pay-your-bid auctions."""

    # --- Bidding dynamics ---
    bid_start: float = 1.50
    costly_bid_start: float = 1.20
    single_good_bid_start: float = 1.80
    zero_reserve_start: float = 0.85
    bid_up: float = 1.50
    bid_down: float = 0.93
    bid_down_full: float = 0.85

    # --- Opening & adaptive ceiling ---
    open_bid: float = 4.00
    open_max_mult: float = 5.00
    probe_max_mult: float = 4.20
    base_max_mult: float = 4.00
    cap_grow: float = 1.35
    cap_shrink: float = 0.88
    cap_limit: float = 60.0
    grow_sell_through: float = 0.68
    shrink_sell_through: float = 0.45

    # --- Economic caps & capital protection ---
    value_share: float = 0.80
    kit_profit_cap: float = 0.85
    budget_share: float = 0.80
    turn1_budget_share: float = 0.58
    probe_capital_floor_frac: float = 0.975  # max 2.5% capital at risk while probing

    def __init__(self) -> None:
        self._bid_level: dict[str, float] = {}
        self._last_bids: dict[str, tuple[int, float]] = {}
        self._cap_mult: float = self.base_max_mult

    def decide_bids(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> list[Bid]:
        """Return sealed bids for this turn's `state.auction_lots`."""
        if state.turn == 1:
            self._bid_level.clear()
            self._last_bids.clear()
            self._cap_mult = self.base_max_mult

        lots = {lot.good: lot for lot in state.auction_lots}
        inv = {k: int(v) for k, v in state.inventory.items()}
        recipes = goods_config.recipes
        probe_only = ctx.is_hyper_crowded and not ctx.proven_hot_market

        # 1. Update adaptive reserve-multiplier ceiling from overall sell-through.
        if ctx.tot_listed_last_turn > 0:
            st = ctx.overall_sell_through
            if st >= self.grow_sell_through:
                self._cap_mult = min(self.cap_limit, self._cap_mult * self.cap_grow)
            elif st < self.shrink_sell_through:
                self._cap_mult = max(self.base_max_mult, self._cap_mult * self.cap_shrink)

        # 2. Adapt per-material bid levels from last turn's fill rate & starving status.
        for mat, (asked, last_price) in self._last_bids.items():
            if asked <= 0:
                continue
            won_q = ctx.own_won_last_turn.get(mat, 0)
            fill = won_q / float(asked)
            starving = any(
                mat in recipes[ctx.recipe_for[g]].inputs
                and inv.get(g, 0) < max(1, ctx.target_stock.get(g, 2) // 2)
                for g in ctx.ranked_goods
                if ctx.need_build.get(g, 0) > 0
            )
            lot = lots.get(mat)
            if fill < 0.80 and starving:
                new_level = last_price * self.bid_up
                r_bids = ctx.rival_bid_prices_last_turn.get(mat)
                if r_bids:
                    min_win = min(r_bids)
                    if min_win > last_price:
                        new_level = min(new_level, min_win * 1.03)
                        new_level = max(new_level, min_win * 1.015)
                self._bid_level[mat] = new_level
            elif fill < 0.80:
                self._bid_level[mat] = last_price
            else:
                decay = self.bid_down_full if fill >= 0.999 else self.bid_down
                new_level = last_price * decay
                if (
                    lot is not None
                    and ctx.lot_growing.get(mat, False)
                    and fill >= 0.95
                    and not ctx.open_active
                    and lot.reserve_price > 1e-6
                ):
                    new_level = min(
                        new_level,
                        max(lot.reserve_price * 1.04, last_price * 0.72),
                    )
                self._bid_level[mat] = new_level

        if state.capital <= 0.01 or not state.auction_lots or not ctx.ranked_goods:
            self._last_bids = {}
            ctx.my_bid_prices = set()
            return []

        # 3. Compute optimal bid price per material (adaptive level + highest-use value cap).
        if ctx.kits_affordable_at_reserve < 45.0:
            init_start = self.costly_bid_start
        elif ctx.single_good_market:
            init_start = self.single_good_bid_start
        else:
            init_start = self.bid_start

        if probe_only:
            active_cap_mult = self.probe_max_mult
        elif ctx.open_active:
            active_cap_mult = self.open_max_mult
        else:
            active_cap_mult = max(self.base_max_mult, self._cap_mult)

        bid_price: dict[str, float] = {}
        for mat in goods_config.raw_materials:
            lot = lots.get(mat)
            if lot is None or lot.quantity <= 0:
                continue
            rprice = float(lot.reserve_price)
            zero_res = rprice <= 1e-6

            # Value cap is driven by the HIGHEST-VALUE active good needing `mat`
            # (never dragged down by cheap low-tier goods).
            val_cap = 0.0
            for g in ctx.ranked_goods:
                if ctx.need_build.get(g, 0) <= 0:
                    continue
                rec = recipes[ctx.recipe_for[g]]
                if mat in rec.inputs:
                    n_in = max(1, sum(int(q) for q in rec.inputs.values()))
                    val_cap = max(val_cap, self.value_share * ctx.ref_price[g] / n_in)

            if val_cap > 0.0 and not zero_res and val_cap < rprice:
                continue

            if zero_res:
                default_zero = max(
                    self.zero_reserve_start,
                    ctx.unit_cost.get(mat, 0.0),
                )
                level = self._bid_level.get(mat, default_zero)
                min_valid = 0.05
                price = max(min_valid, level)
                if val_cap > 0.0:
                    price = min(price, max(val_cap, min_valid))
            else:
                level = self._bid_level.get(mat, rprice * init_start)
                if ctx.open_active:
                    level = max(level, rprice * self.open_bid)
                min_valid = rprice * 1.001
                price = max(min_valid, level)
                price = min(price, rprice * active_cap_mult)
                if val_cap > 0.0:
                    price = min(price, max(val_cap, min_valid))

            if isfinite(price) and price >= rprice:
                bid_price[mat] = price

        # 4. Compute safe budget: strict capital-protection floor while probing hyper-crowded rooms.
        if probe_only:
            risk_budget = max(
                0.0,
                float(state.capital) - ctx.starting_capital * self.probe_capital_floor_frac,
            )
            rem_budget = min(float(state.capital) * 0.08, risk_budget)
        else:
            active_budget_share = (
                self.turn1_budget_share
                if state.turn == 1 and ctx.open_active
                else self.budget_share
            )
            rem_budget = max(0.0, float(state.capital) * active_budget_share)

        if rem_budget <= 1e-6:
            self._last_bids = {}
            ctx.my_bid_prices = set()
            return []

        # 5. Kit-aligned budget allocation in descending recipe priority order.
        # Per-good kit profitability check skips unprofitable low-tier goods without
        # penalizing bids for high-margin goods sharing the same raw materials.
        rem_raw = {m: max(0, inv.get(m, 0)) for m in goods_config.raw_materials}
        rem_lot = {m: int(lot.quantity) for m, lot in lots.items()}
        buy_qty: dict[str, int] = {m: 0 for m in goods_config.raw_materials}

        for g in ctx.ranked_goods:
            target_units = ctx.need_build.get(g, 0)
            if target_units <= 0:
                continue
            rec = recipes[ctx.recipe_for[g]]
            if not rec.inputs:
                continue

            # Economic kit check: skip goods whose full kit at current bid prices
            # exceeds `kit_profit_cap * ref_price[g]`.
            full_kit_cost = sum(
                bid_price.get(m, ctx.unit_cost.get(m, float(lots[m].reserve_price) if m in lots else 0.0))
                * int(amt)
                for m, amt in rec.inputs.items()
            )
            if full_kit_cost > self.kit_profit_cap * ctx.ref_price[g]:
                continue

            for _ in range(target_units):
                kit_spend = 0.0
                feasible = True
                for m, amt in rec.inputs.items():
                    amt_i = int(amt)
                    short = max(0, amt_i - rem_raw.get(m, 0))
                    if short > 0:
                        if m not in bid_price or short > rem_lot.get(m, 0):
                            feasible = False
                            break
                        kit_spend += short * bid_price[m]
                if not feasible or kit_spend > rem_budget + 1e-9:
                    break
                rem_budget -= kit_spend
                for m, amt in rec.inputs.items():
                    amt_i = int(amt)
                    from_inv = min(rem_raw.get(m, 0), amt_i)
                    rem_raw[m] = rem_raw.get(m, 0) - from_inv
                    short = amt_i - from_inv
                    if short > 0:
                        buy_qty[m] = buy_qty.get(m, 0) + short
                        rem_lot[m] = rem_lot.get(m, 0) - short

        # 6. Emit final bids & store metadata for next turn's fill-rate learning.
        bids: list[Bid] = []
        self._last_bids = {}
        my_prices: set[float] = set()
        for m, qty in buy_qty.items():
            if qty <= 0 or m not in bid_price:
                continue
            p = round(float(bid_price[m]), 4)
            if isfinite(p) and p >= float(lots[m].reserve_price):
                bids.append(Bid(m, int(qty), p))
                self._last_bids[m] = (int(qty), p)
                my_prices.add(p)

        ctx.my_bid_prices = my_prices
        return bids
