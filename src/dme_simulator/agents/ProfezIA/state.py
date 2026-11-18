"""Shared market state and incremental history tracking (Owner: Persona 3).

Updated once at the start of every turn by `agent.py` before calling
`bidding`, `production`, and `pricing`.

Backward-compatibility contract (used by Persona 1 & Persona 2):
- `turn`, `team_id`
- `new_auction_txs`, `new_sales_txs`
- `reserve_price_history`, `lot_quantity_history`
- `avg_material_cost`, `est_sale_price`
- `planned_bids`, `planned_assembly`
- `estimate_recipe_cost(recipe_name, goods_config)`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log1p

from dme_simulator.agents import (
    AssemblyOrder,
    AuctionResult,
    Bid,
    Observation,
    SalesPrice,
)
from dme_simulator.config import GoodsConfig


@dataclass
class MarketContext:
    """Shared read-only market signals computed incrementally from observations."""

    turn: int = 0
    team_id: str | None = None

    # Only transactions added since the previous turn (avoids O(T^2) rescans).
    new_auction_txs: list[AuctionResult] = field(default_factory=list)
    new_sales_txs: list[SalesPrice] = field(default_factory=list)
    _seen_auction_count: int = 0
    _seen_sales_count: int = 0

    # Per-material auction history (appended every turn).
    reserve_price_history: dict[str, list[float]] = field(default_factory=dict)
    lot_quantity_history: dict[str, list[int]] = field(default_factory=dict)

    # Exponentially smoothed estimates updated from completed transactions.
    avg_material_cost: dict[str, float] = field(default_factory=dict)
    est_sale_price: dict[str, float] = field(default_factory=dict)

    # Current-turn upstream plans, populated sequentially by agent.py.
    planned_bids: list[Bid] = field(default_factory=list)
    planned_assembly: list[AssemblyOrder] = field(default_factory=list)

    # --- Additive intelligence fields (Persona 3) ---
    # Own exact per-turn accounting inferred from transaction deltas + inventory.
    own_won_last_turn: dict[str, int] = field(default_factory=dict)
    own_spend_last_turn: dict[str, float] = field(default_factory=dict)
    own_sold_last_turn: dict[str, int] = field(default_factory=dict)
    own_revenue_last_turn: dict[str, float] = field(default_factory=dict)
    own_produced_last_turn: dict[str, int] = field(default_factory=dict)

    # Own smoothed sales velocity and demand-pressure streaks per finished good.
    own_sold_ema: dict[str, float] = field(default_factory=dict)
    sales_count_by_good: dict[str, int] = field(default_factory=dict)
    own_last_sale_price: dict[str, float] = field(default_factory=dict)
    own_stockout_streak: dict[str, int] = field(default_factory=dict)
    own_unsold_streak: dict[str, int] = field(default_factory=dict)

    # Auction execution quality and automatic production capacity detection.
    auction_fill_rate: dict[str, float] = field(default_factory=dict)
    estimated_capacity_per_turn: int | None = None

    # Public competitor and supply-scarcity intelligence.
    active_competitor_count: int = 0
    competitor_finished_inventory: dict[str, int] = field(default_factory=dict)
    competitor_raw_inventory: dict[str, int] = field(default_factory=dict)
    lot_scarcity: dict[str, float] = field(default_factory=dict)
    rival_capital_delta: dict[str, float] = field(default_factory=dict)
    market_demand_heat: float = 1.0

    # Downstream economic signals (populated in state.py and refined by production.py).
    recipe_margins: dict[str, float] = field(default_factory=dict)
    material_shadow_value: dict[str, float] = field(default_factory=dict)
    target_material_demand: dict[str, int] = field(default_factory=dict)

    # Internal previous-turn snapshots for mass-balance & capacity inference.
    _prev_inventory: dict[str, int] | None = None
    _prev_reserve_prices: dict[str, float] = field(default_factory=dict)
    _max_reserve_prices: dict[str, float] = field(default_factory=dict)
    _prev_public_capital: dict[str, float] = field(default_factory=dict)
    _prev_comp_finished: dict[str, int] = field(default_factory=dict)
    _rival_inflow_ema: float = 0.0

    def update(self, state: Observation, goods_config: GoodsConfig) -> None:
        """Advance shared market estimates using the latest turn observation."""
        prev_bids = self.planned_bids
        prev_assembly = self.planned_assembly

        self.turn = state.turn
        self.planned_bids = []
        self.planned_assembly = []

        # 1. Slice only newly arrived transactions since last turn.
        self.new_auction_txs = state.auction_transactions[self._seen_auction_count :]
        self._seen_auction_count = len(state.auction_transactions)

        self.new_sales_txs = state.sales_transactions[self._seen_sales_count :]
        self._seen_sales_count = len(state.sales_transactions)

        # 2. Identify our own team_id as early as possible.
        if self.team_id is None:
            self.team_id = self._infer_team_id(
                state, self.new_auction_txs, self.new_sales_txs
            )

        # 3. Record current auction lots, reserve history, and lot scarcity.
        current_reserves: dict[str, float] = {}
        for lot in state.auction_lots:
            current_reserves[lot.good] = lot.reserve_price
            res_hist = self.reserve_price_history.setdefault(lot.good, [])
            qty_hist = self.lot_quantity_history.setdefault(lot.good, [])
            prev_qty = qty_hist[-1] if qty_hist else None

            res_hist.append(lot.reserve_price)
            qty_hist.append(lot.quantity)
            self.avg_material_cost.setdefault(lot.good, lot.reserve_price)

            max_res = max(self._max_reserve_prices.get(lot.good, 0.0), lot.reserve_price)
            self._max_reserve_prices[lot.good] = max_res
            price_pressure = lot.reserve_price / max_res if max_res > 0.0 else 0.5
            if prev_qty is not None and prev_qty > 0:
                depletion = max(0.0, min(1.0, 1.0 - lot.quantity / prev_qty))
                scarcity = 0.65 * price_pressure + 0.35 * depletion
            else:
                scarcity = price_pressure
            self.lot_scarcity[lot.good] = max(0.0, min(1.0, scarcity))

        # 4. Update running material cost & finished-good sale price estimates.
        alpha_price = 0.35
        for tx in self.new_auction_txs:
            prev = self.avg_material_cost.get(tx.good, tx.price)
            self.avg_material_cost[tx.good] = (1.0 - alpha_price) * prev + alpha_price * tx.price

        for sale in self.new_sales_txs:
            prev = self.est_sale_price.get(sale.good, 0.0)
            if prev <= 0.0:
                self.est_sale_price[sale.good] = sale.price
            else:
                self.est_sale_price[sale.good] = (
                    (1.0 - alpha_price) * prev + alpha_price * sale.price
                )

        # Ensure every raw material and finished good has a fallback key.
        for raw in goods_config.raw_materials:
            self.avg_material_cost.setdefault(raw, 0.0)
            self.auction_fill_rate.setdefault(raw, 0.60)
            self.lot_scarcity.setdefault(raw, 0.5)
        for good in goods_config.finished_goods:
            self.est_sale_price.setdefault(good, 0.0)
            self.sales_count_by_good.setdefault(good, 0)
            self.own_sold_ema.setdefault(good, 0.0)
            self.own_stockout_streak.setdefault(good, 0)
            self.own_unsold_streak.setdefault(good, 0)

        # 5. Compute own per-turn mass balance, fill rates, and capacity detection.
        self._update_own_accounting(state, goods_config, prev_bids, prev_assembly)

        # 6. Compute active competitor inventory intelligence & market demand heat.
        self._update_competitor_intelligence(state, goods_config)

        # 7. Refresh per-recipe estimated margins and baseline material shadow values.
        self._update_recipe_economics(goods_config)

        # 8. Save current snapshot for next turn's delta calculations.
        self._prev_inventory = dict(state.inventory)
        self._prev_reserve_prices = current_reserves
        self._prev_public_capital = dict(state.public_capital)
        self._prev_comp_finished = dict(self.competitor_finished_inventory)

    def _update_own_accounting(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        prev_bids: list[Bid],
        prev_assembly: list[AssemblyOrder],
    ) -> None:
        own_won: dict[str, int] = {}
        own_spend: dict[str, float] = {}
        own_sold: dict[str, int] = {}
        own_rev: dict[str, float] = {}

        # Identify which transactions belong to us.
        if self.team_id is not None:
            for tx in self.new_auction_txs:
                if tx.buyer == self.team_id:
                    own_won[tx.good] = own_won.get(tx.good, 0) + tx.quantity
                    own_spend[tx.good] = own_spend.get(tx.good, 0.0) + tx.price * tx.quantity
            for sale in self.new_sales_txs:
                if sale.seller == self.team_id:
                    own_sold[sale.good] = own_sold.get(sale.good, 0) + sale.quantity
                    own_rev[sale.good] = own_rev.get(sale.good, 0.0) + sale.price * sale.quantity
                    self.own_last_sale_price[sale.good] = sale.price
        else:
            # Fallback if team_id is still ambiguous: accept only if a single actor appears.
            buyers = {tx.buyer for tx in self.new_auction_txs}
            if len(buyers) == 1:
                for tx in self.new_auction_txs:
                    own_won[tx.good] = own_won.get(tx.good, 0) + tx.quantity
                    own_spend[tx.good] = own_spend.get(tx.good, 0.0) + tx.price * tx.quantity
            sellers = {s.seller for s in self.new_sales_txs}
            if len(sellers) == 1:
                for sale in self.new_sales_txs:
                    own_sold[sale.good] = own_sold.get(sale.good, 0) + sale.quantity
                    own_rev[sale.good] = own_rev.get(sale.good, 0.0) + sale.price * sale.quantity
                    self.own_last_sale_price[sale.good] = sale.price

        self.own_won_last_turn = own_won
        self.own_spend_last_turn = own_spend
        self.own_sold_last_turn = own_sold
        self.own_revenue_last_turn = own_rev

        for good, qty in own_sold.items():
            self.sales_count_by_good[good] = self.sales_count_by_good.get(good, 0) + qty

        if self._prev_inventory is None:
            self.own_produced_last_turn = {}
            return

        # Exact finished-good production via mass balance:
        # inv_t[g] = inv_{t-1}[g] + produced_{t-1}[g] - sold_{t-1}[g]
        produced: dict[str, int] = {}
        alpha_vel = 0.35
        for good in goods_config.finished_goods:
            prev_qty = self._prev_inventory.get(good, 0)
            curr_qty = state.inventory.get(good, 0)
            sold_qty = own_sold.get(good, 0)
            prod_qty = max(0, curr_qty - prev_qty + sold_qty)
            if prod_qty > 0:
                produced[good] = prod_qty

            available_for_sale = prev_qty + prod_qty
            if available_for_sale > 0:
                prev_ema = self.own_sold_ema.get(good, 0.0)
                if prev_ema <= 0.0:
                    self.own_sold_ema[good] = float(sold_qty)
                else:
                    self.own_sold_ema[good] = (
                        (1.0 - alpha_vel) * prev_ema + alpha_vel * sold_qty
                    )

                if sold_qty >= available_for_sale or (sold_qty > 0 and curr_qty == 0):
                    self.own_stockout_streak[good] = self.own_stockout_streak.get(good, 0) + 1
                    self.own_unsold_streak[good] = 0
                elif sold_qty == 0 and curr_qty > 0:
                    self.own_unsold_streak[good] = self.own_unsold_streak.get(good, 0) + 1
                    self.own_stockout_streak[good] = 0
                else:
                    self.own_stockout_streak[good] = 0
                    self.own_unsold_streak[good] = 0
        self.own_produced_last_turn = produced

        # Update auction fill-rate EMA from eligible previous-turn bids.
        eligible_bid_qty: dict[str, int] = {}
        for bid in prev_bids:
            reserve = self._prev_reserve_prices.get(bid.good, 0.0)
            if bid.quantity > 0 and bid.price_per_unit >= reserve - 1e-9:
                eligible_bid_qty[bid.good] = eligible_bid_qty.get(bid.good, 0) + bid.quantity
        alpha_fill = 0.35
        for good, req_qty in eligible_bid_qty.items():
            obs_fill = min(1.0, own_won.get(good, 0) / req_qty)
            prev_fill = self.auction_fill_rate.get(good, 0.60)
            self.auction_fill_rate[good] = (
                (1.0 - alpha_fill) * prev_fill + alpha_fill * obs_fill
            )

        # Automatically detect production.capacity_per_turn if our assembly was capped.
        if prev_assembly and self.team_id is not None:
            sim_raw = {
                m: self._prev_inventory.get(m, 0) + own_won.get(m, 0)
                for m in goods_config.raw_materials
            }
            max_buildable = 0
            for order in prev_assembly:
                recipe = goods_config.recipes.get(order.recipe_name)
                if recipe is None or order.quantity <= 0 or not recipe.inputs:
                    continue
                can_make = min(
                    sim_raw.get(mat, 0) // units for mat, units in recipe.inputs.items()
                )
                built = min(order.quantity, can_make)
                if built > 0:
                    max_buildable += built
                    for mat, units in recipe.inputs.items():
                        sim_raw[mat] -= units * built
            actual_total = sum(produced.values())
            if 0 < actual_total < max_buildable:
                if self.estimated_capacity_per_turn is None:
                    self.estimated_capacity_per_turn = actual_total
                else:
                    self.estimated_capacity_per_turn = max(
                        self.estimated_capacity_per_turn, actual_total
                    )

    def _update_competitor_intelligence(
        self, state: Observation, goods_config: GoodsConfig
    ) -> None:
        comp_finished = {g: 0 for g in goods_config.finished_goods}
        comp_raw = {m: 0 for m in goods_config.raw_materials}
        rival_deltas: dict[str, float] = {}
        active_rivals = 0
        positive_inflow = 0.0

        for tid, inv in state.public_inventories.items():
            if tid == self.team_id:
                continue
            cap = state.public_capital.get(tid, 0.0)
            if cap < 0.0:
                continue
            active_rivals += 1
            if tid in self._prev_public_capital:
                delta = cap - self._prev_public_capital[tid]
                rival_deltas[tid] = delta
                if delta > 0.0:
                    positive_inflow += delta
            for g in goods_config.finished_goods:
                comp_finished[g] += inv.get(g, 0)
            for m in goods_config.raw_materials:
                comp_raw[m] += inv.get(m, 0)

        self.active_competitor_count = active_rivals
        self.competitor_finished_inventory = comp_finished
        self.competitor_raw_inventory = comp_raw
        self.rival_capital_delta = rival_deltas

        # Track market demand heat from positive rival cash inflows + own sales velocity.
        own_rev_turn = sum(self.own_revenue_last_turn.values())
        total_flow = positive_inflow + own_rev_turn
        if self._prev_public_capital:
            if self._rival_inflow_ema <= 0.0:
                self._rival_inflow_ema = total_flow
                self.market_demand_heat = 1.0
            else:
                ratio = total_flow / max(1.0, self._rival_inflow_ema)
                raw_heat = max(0.3, min(2.0, ratio))
                self.market_demand_heat = 0.65 * self.market_demand_heat + 0.35 * raw_heat
                self._rival_inflow_ema = 0.75 * self._rival_inflow_ema + 0.25 * total_flow

    def _update_recipe_economics(self, goods_config: GoodsConfig) -> None:
        """Compute baseline per-recipe margins and per-material shadow values."""
        margins: dict[str, float] = {}
        baseline_shadow: dict[str, float] = {m: 0.0 for m in goods_config.raw_materials}

        for r_name, recipe in goods_config.recipes.items():
            rev = self.estimate_good_revenue(recipe.output_good, goods_config)
            cost = self.estimate_recipe_cost(r_name, goods_config)
            margin = rev - cost
            margins[r_name] = margin

            total_inputs = sum(recipe.inputs.values())
            if total_inputs <= 0 or rev <= 0.0:
                continue
            # Value per input unit proportional to material's cost share (or uniform if zero cost).
            for mat, units in recipe.inputs.items():
                if cost > 1e-6:
                    mat_share = (self.avg_material_cost.get(mat, 0.0) * units) / cost
                    implied_val = (rev * mat_share) / units
                else:
                    implied_val = rev / total_inputs
                if implied_val > baseline_shadow.get(mat, 0.0):
                    baseline_shadow[mat] = implied_val

        self.recipe_margins = margins
        for mat, val in baseline_shadow.items():
            # Keep any higher/refined shadow value set by production if already initialized,
            # while refreshing the floor each turn.
            self.material_shadow_value[mat] = val

    def estimate_recipe_cost(self, recipe_name: str, goods_config: GoodsConfig) -> float:
        """Return estimated raw-material unit cost for one unit of `recipe_name`."""
        recipe = goods_config.recipes[recipe_name]
        return sum(
            self.avg_material_cost.get(mat, 0.0) * qty
            for mat, qty in recipe.inputs.items()
        )

    def estimate_good_revenue(self, good: str, goods_config: GoodsConfig) -> float:
        """Return expected unit revenue for `good`, with a cost-scaled cold-start prior."""
        observed_price = self.est_sale_price.get(good, 0.0)
        if self.sales_count_by_good.get(good, 0) > 0 and observed_price > 0.0:
            return observed_price

        # Also use observed market price if `observe_all_transactions` populated est_sale_price.
        if observed_price > 0.0:
            return observed_price

        # Cold-start prior: scale cheapest recipe cost by observed market markup ratio.
        recipe_costs: dict[str, float] = {}
        for r_name, recipe in goods_config.recipes.items():
            cost = self.estimate_recipe_cost(r_name, goods_config)
            prev = recipe_costs.get(recipe.output_good)
            if prev is None or cost < prev:
                recipe_costs[recipe.output_good] = cost

        ratios: list[float] = []
        for g, price in self.est_sale_price.items():
            c = recipe_costs.get(g, 0.0)
            if price > 0.0 and c > 1e-6:
                ratios.append(price / c)
        if ratios:
            prior_ratio = max(2.0, min(12.0, sum(ratios) / len(ratios)))
        else:
            prior_ratio = 4.0

        good_cost = recipe_costs.get(good, 0.0)
        if good_cost > 0.0:
            positive_costs = [c for c in recipe_costs.values() if c > 0.0]
            avg_cost = sum(positive_costs) / len(positive_costs) if positive_costs else good_cost
            tier_bonus = 1.0 + 0.15 * log1p(good_cost / max(1e-6, avg_cost))
            return good_cost * prior_ratio * tier_bonus
        return 100.0

    @staticmethod
    def _infer_team_id(
        state: Observation,
        new_auction_txs: list[AuctionResult],
        new_sales_txs: list[SalesPrice],
    ) -> str | None:
        matches = [
            tid
            for tid, inv in state.public_inventories.items()
            if inv == state.inventory
            and abs(state.public_capital.get(tid, -1.0) - state.capital) < 1e-6
        ]
        if len(matches) == 1:
            return matches[0]

        match_set = set(matches) if matches else set(state.public_inventories)
        tx_actors = {tx.buyer for tx in new_auction_txs} | {
            sale.seller for sale in new_sales_txs
        }
        narrowed = match_set & tx_actors
        if len(narrowed) == 1:
            return next(iter(narrowed))
        return None
