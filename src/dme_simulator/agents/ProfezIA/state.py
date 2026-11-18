"""Shared market state, incremental accounting, and unified economic model for ProfezIA.

Updated once at the start of every turn by `agent.py` before calling
`bidding`, `production`, and `pricing`. Provides a single coherent source of
truth for costs, reference prices, demand rates, target stocks, and rival signals
so all three phases stay tightly synchronized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

    # --- Pricing & demand parameters ---
    prior_markup: float = 7.0
    single_good_prior_markup: float = 5.0
    costly_prior_markup: float = 4.5
    explore_turns: int = 6
    step_up: float = 0.07
    step_down: float = 0.15
    up_thresh: float = 0.90
    min_markup: float = 1.30
    floor_patience: int = 3
    floor_relax: float = 0.85
    floor_min: float = 0.55

    # --- Stock & opening parameters ---
    min_stock: int = 2
    max_stock: int = 200
    cover_turns: float = 3.0
    open_turns: int = 5
    open_stock: int = 6

    # --- Public turn state ---
    turn: int = 0
    team_id: str | None = None
    starting_capital: float = 12000.0
    nominal_reserve_regime: bool = False
    kits_affordable_at_reserve: float = 200.0
    single_good_market: bool = False
    supply_per_team: float = 10.0
    is_hyper_crowded: bool = False
    proven_hot_market: bool = False
    open_active: bool = False

    # Incremental transaction slices
    new_auction_txs: list[AuctionResult] = field(default_factory=list)
    new_sales_txs: list[SalesPrice] = field(default_factory=list)
    _seen_auction_count: int = 0
    _seen_sales_count: int = 0
    _id_candidates: set[str] | None = None

    # Current-turn plans populated sequentially by agent.py
    planned_bids: list[Bid] = field(default_factory=list)
    planned_assembly: list[AssemblyOrder] = field(default_factory=list)

    # Own last-turn accounting
    own_won_last_turn: dict[str, int] = field(default_factory=dict)
    own_spend_last_turn: dict[str, float] = field(default_factory=dict)
    own_sold_prices_last_turn: dict[str, list[float]] = field(default_factory=dict)
    rival_sold_prices_last_turn: dict[str, list[float]] = field(default_factory=dict)
    rival_bid_prices_last_turn: dict[str, list[float]] = field(default_factory=dict)

    # Material cost & supplier lot tracking
    reserve_price: dict[str, float] = field(default_factory=dict)
    unit_cost: dict[str, float] = field(default_factory=dict)
    prev_lot_qty: dict[str, int] = field(default_factory=dict)
    lot_growing: dict[str, bool] = field(default_factory=dict)

    # Cheapest recipe per finished good
    recipe_for: dict[str, str] = field(default_factory=dict)
    good_cost: dict[str, float] = field(default_factory=dict)
    good_reserve_cost: dict[str, float] = field(default_factory=dict)

    # Unified pricing & demand state
    ref_price: dict[str, float] = field(default_factory=dict)
    sales_rate: dict[str, float] = field(default_factory=dict)
    floor_mult: dict[str, float] = field(default_factory=dict)
    zero_streak: dict[str, int] = field(default_factory=dict)
    total_sold: dict[str, int] = field(default_factory=dict)
    init_endowment: dict[str, int] = field(default_factory=dict)
    last_ladder: dict[str, list[float]] = field(default_factory=dict)
    my_list_prices: set[float] = field(default_factory=set)
    my_bid_prices: set[float] = field(default_factory=set)

    # Overall sell-through & capacity detection
    tot_sold_last_turn: int = 0
    tot_listed_last_turn: int = 0
    overall_sell_through: float = 0.0
    estimated_capacity_per_turn: int | None = None

    # Rival intelligence from public inventories
    rival_stock: dict[str, int] = field(default_factory=dict)
    rival_holders: dict[str, int] = field(default_factory=dict)
    rival_sold: dict[str, int] = field(default_factory=dict)
    active_team_count: int = 1

    # Unified turn production targets (shared by bidding, production, pricing)
    ranked_goods: list[str] = field(default_factory=list)
    target_stock: dict[str, int] = field(default_factory=dict)
    need_build: dict[str, int] = field(default_factory=dict)

    # Internal previous-turn snapshots
    _prev_inventory: dict[str, int] | None = None
    _prev_public_inv: dict[str, dict[str, int]] = field(default_factory=dict)
    _initialized: bool = False

    def update(self, state: Observation, goods_config: GoodsConfig) -> None:
        """Advance shared market state using the latest turn observation."""
        prev_assembly = self.planned_assembly
        self.turn = state.turn
        self.planned_bids = []
        self.planned_assembly = []

        if not self._initialized or state.turn == 1:
            self._reset_for_new_game(state, goods_config)

        # 1. Slice only newly arrived transactions since last turn.
        self.new_auction_txs = state.auction_transactions[self._seen_auction_count :]
        self._seen_auction_count = len(state.auction_transactions)
        self.new_sales_txs = state.sales_transactions[self._seen_sales_count :]
        self._seen_sales_count = len(state.sales_transactions)

        # 2. Identify own team_id cleanly (handles observe_all_transactions=True).
        self._identify(state)

        # 3. Digest new auction & sales transactions (own vs rival).
        own_won: dict[str, int] = {}
        own_spend: dict[str, float] = {}
        rival_bids: dict[str, list[float]] = {}
        for tx in self.new_auction_txs:
            if tx.quantity <= 0:
                continue
            if self._is_mine(tx.buyer, tx.price, self.my_bid_prices):
                own_won[tx.good] = own_won.get(tx.good, 0) + tx.quantity
                own_spend[tx.good] = own_spend.get(tx.good, 0.0) + tx.price * tx.quantity
            else:
                rival_bids.setdefault(tx.good, []).append(float(tx.price))

        own_sold_prices: dict[str, list[float]] = {}
        rival_sold_prices: dict[str, list[float]] = {}
        for tx in self.new_sales_txs:
            if tx.quantity <= 0:
                continue
            if self._is_mine(tx.seller, tx.price, self.my_list_prices):
                own_sold_prices.setdefault(tx.good, []).extend([float(tx.price)] * tx.quantity)
            else:
                rival_sold_prices.setdefault(tx.good, []).extend([float(tx.price)] * tx.quantity)

        self.own_won_last_turn = own_won
        self.own_spend_last_turn = own_spend
        self.own_sold_prices_last_turn = own_sold_prices
        self.rival_sold_prices_last_turn = rival_sold_prices
        self.rival_bid_prices_last_turn = rival_bids

        # 4. Update material reserves, lot growth signals, and actual paid unit_cost EMA.
        lots = {lot.good: lot for lot in state.auction_lots}
        for raw in goods_config.raw_materials:
            lot = lots.get(raw)
            if lot is not None:
                prev_q = self.prev_lot_qty.get(raw)
                self.lot_growing[raw] = prev_q is not None and lot.quantity > prev_q
                self.prev_lot_qty[raw] = int(lot.quantity)
                self.reserve_price[raw] = float(lot.reserve_price)
            else:
                self.lot_growing[raw] = False
                self.reserve_price.setdefault(raw, 1.0)

            won_q = own_won.get(raw, 0)
            if won_q > 0:
                avg_paid = own_spend[raw] / float(won_q)
                if raw in self.unit_cost:
                    self.unit_cost[raw] = 0.70 * self.unit_cost[raw] + 0.30 * avg_paid
                else:
                    self.unit_cost[raw] = avg_paid

        # 5. Compute cheapest recipe per finished good at actual paid cost & reserve cost.
        self._update_recipe_costs(goods_config)

        # 6. Detect production.capacity_per_turn automatically from mass balance.
        self._detect_capacity(state, goods_config, prev_assembly, own_won, own_sold_prices)

        # 7. Observe rivals from public inventories (stock, holders, inferred sales).
        self._update_rival_signals(state, goods_config, rival_sold_prices)

        # 8. Update ladder-driven reference prices and sales rates.
        self._update_prices_and_demand(state, goods_config, own_sold_prices, rival_sold_prices)

        # 9. Compute unified target stocks, ranked profitable goods, and build needs.
        self._plan_targets_and_needs(state, goods_config)

        # 10. Save snapshots for next turn's delta calculations.
        self._prev_inventory = {k: int(v) for k, v in state.inventory.items()}
        self._prev_public_inv = {
            tid: {k: int(v) for k, v in inv.items()}
            for tid, inv in state.public_inventories.items()
        }

    # ------------------------------------------------------------------
    # Initialization & identification
    # ------------------------------------------------------------------

    def _reset_for_new_game(self, state: Observation, goods_config: GoodsConfig) -> None:
        self._initialized = True
        self.team_id = None
        self._id_candidates = None
        self._seen_auction_count = 0
        self._seen_sales_count = 0
        self.starting_capital = max(1.0, float(state.capital))
        self.single_good_market = len(goods_config.finished_goods) == 1
        self.unit_cost.clear()
        self.reserve_price.clear()
        self.prev_lot_qty.clear()
        self.lot_growing.clear()
        self.recipe_for.clear()
        self.good_cost.clear()
        self.good_reserve_cost.clear()
        self.ref_price.clear()
        self.sales_rate.clear()
        self.floor_mult.clear()
        self.zero_streak.clear()
        self.total_sold.clear()
        self.last_ladder.clear()
        self.my_list_prices.clear()
        self.my_bid_prices.clear()
        self.estimated_capacity_per_turn = None
        self._prev_inventory = None
        self._prev_public_inv.clear()
        self.tot_sold_last_turn = 0
        self.tot_listed_last_turn = 0
        self.overall_sell_through = 0.0
        self.proven_hot_market = False
        self.is_hyper_crowded = False
        self.supply_per_team = 10.0

        self.init_endowment = {
            g: max(0, int(state.inventory.get(g, 0)))
            for g in goods_config.finished_goods
        }

        lots = {lot.good: float(lot.reserve_price) for lot in state.auction_lots}
        total_init_lots = sum(max(0, int(lot.quantity)) for lot in state.auction_lots)
        recipe_res_costs: list[float] = []
        for recipe in goods_config.recipes.values():
            c = sum(lots.get(m, 0.0) * int(q) for m, q in recipe.inputs.items())
            recipe_res_costs.append(c)

        max_res_cost = max(recipe_res_costs) if recipe_res_costs else 0.0
        avg_res_cost = (
            sum(recipe_res_costs) / len(recipe_res_costs) if recipe_res_costs else 0.0
        )
        # Only treat zero/near-zero reserves (e.g. tutorial_pricing, priv_zero_reserve) as
        # nominal-cost regimes; never misclassify scaled_market or default_extended.
        self.nominal_reserve_regime = (
            (max_res_cost <= 0.05 and self.starting_capital >= 500.0)
            or (
                max_res_cost <= 3.0
                and self.starting_capital >= 1000.0
                and total_init_lots >= 300
            )
        )
        self.kits_affordable_at_reserve = (
            self.starting_capital / avg_res_cost if avg_res_cost > 1e-6 else 1000.0
        )

    def _identify(self, state: Observation) -> None:
        if self.team_id is not None:
            return
        mine_nz = {k: int(v) for k, v in state.inventory.items() if v}
        matches = {
            tid
            for tid, inv in state.public_inventories.items()
            if abs(state.public_capital.get(tid, -1e18) - state.capital) < 1e-5
            and {k: int(v) for k, v in inv.items() if v} == mine_nz
        }
        if self._id_candidates is None:
            self._id_candidates = matches
        else:
            narrowed = self._id_candidates & matches
            self._id_candidates = narrowed if narrowed else matches

        if len(self._id_candidates) == 1:
            self.team_id = next(iter(self._id_candidates))
            return

        if self.my_list_prices:
            for tx in self.new_sales_txs:
                if tx.seller in self._id_candidates and round(float(tx.price), 4) in self.my_list_prices:
                    self.team_id = tx.seller
                    return
        if self.my_bid_prices:
            for tx in self.new_auction_txs:
                if tx.buyer in self._id_candidates and round(float(tx.price), 4) in self.my_bid_prices:
                    self.team_id = tx.buyer
                    return

    def _is_mine(self, actor: str, price: float, my_prices: set[float]) -> bool:
        if self.team_id is not None:
            return actor == self.team_id
        if my_prices:
            return round(float(price), 4) in my_prices
        return True

    # ------------------------------------------------------------------
    # Cost, capacity, rival signals
    # ------------------------------------------------------------------

    def _update_recipe_costs(self, goods_config: GoodsConfig) -> None:
        mat_cost = {
            m: self.unit_cost.get(m, self.reserve_price.get(m, 1.0))
            for m in goods_config.raw_materials
        }
        recipe_for: dict[str, str] = {}
        good_cost: dict[str, float] = {}
        good_res_cost: dict[str, float] = {}

        for name, rec in goods_config.recipes.items():
            g = rec.output_good
            c_paid = sum(mat_cost.get(m, 1.0) * int(q) for m, q in rec.inputs.items())
            c_res = sum(self.reserve_price.get(m, 0.0) * int(q) for m, q in rec.inputs.items())
            if g not in good_cost or c_paid < good_cost[g]:
                good_cost[g] = max(c_paid, 1e-6)
                good_res_cost[g] = max(c_res, 0.0)
                recipe_for[g] = name

        self.recipe_for = recipe_for
        self.good_cost = good_cost
        self.good_reserve_cost = good_res_cost

    def _detect_capacity(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        prev_assembly: list[AssemblyOrder],
        own_won: dict[str, int],
        own_sold_prices: dict[str, list[float]],
    ) -> None:
        if self._prev_inventory is None or not prev_assembly:
            return
        sim_raw = {
            m: self._prev_inventory.get(m, 0) + own_won.get(m, 0)
            for m in goods_config.raw_materials
        }
        max_buildable = 0
        for order in prev_assembly:
            rec = goods_config.recipes.get(order.recipe_name)
            if rec is None or order.quantity <= 0 or not rec.inputs:
                continue
            can_make = min(
                sim_raw.get(m, 0) // int(u) for m, u in rec.inputs.items()
            )
            built = min(int(order.quantity), can_make)
            if built > 0:
                max_buildable += built
                for m, u in rec.inputs.items():
                    sim_raw[m] -= int(u) * built

        actual_total = 0
        for g in goods_config.finished_goods:
            prev_g = self._prev_inventory.get(g, 0)
            curr_g = int(state.inventory.get(g, 0))
            sold_g = len(own_sold_prices.get(g, []))
            actual_total += max(0, curr_g - prev_g + sold_g)

        if 0 < actual_total < max_buildable:
            if self.estimated_capacity_per_turn is None:
                self.estimated_capacity_per_turn = actual_total
            else:
                self.estimated_capacity_per_turn = max(
                    self.estimated_capacity_per_turn, actual_total
                )

    def _update_rival_signals(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        rival_sold_prices: dict[str, list[float]],
    ) -> None:
        self.active_team_count = max(1, len(state.public_capital))
        mine = state.inventory
        prev_mine = self._prev_inventory or {}
        rival_stock: dict[str, int] = {}
        rival_holders: dict[str, int] = {}
        rival_sold: dict[str, int] = {}

        for g in goods_config.finished_goods:
            tot = 0
            holders = 0
            drop = 0
            for tid, inv in state.public_inventories.items():
                q = int(inv.get(g, 0))
                tot += q
                if q > 0:
                    holders += 1
                prev_q = self._prev_public_inv.get(tid, {}).get(g, 0)
                if prev_q > q:
                    drop += prev_q - q
            my_q = int(mine.get(g, 0))
            my_drop = max(0, prev_mine.get(g, 0) - my_q)
            rival_stock[g] = max(0, tot - my_q)
            rival_holders[g] = max(0, holders - (1 if my_q > 0 else 0))
            rival_sold[g] = max(0, drop - my_drop, len(rival_sold_prices.get(g, [])))

        self.rival_stock = rival_stock
        self.rival_holders = rival_holders
        self.rival_sold = rival_sold

    # ------------------------------------------------------------------
    # Price & demand learning
    # ------------------------------------------------------------------

    def _update_prices_and_demand(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        own_sold_prices: dict[str, list[float]],
        rival_sold_prices: dict[str, list[float]],
    ) -> None:
        early = self.turn <= self.explore_turns
        tot_sold = 0
        tot_listed = 0

        input_counts = {
            g: max(1, sum(int(q) for q in goods_config.recipes[r].inputs.values()))
            for g, r in self.recipe_for.items()
        }
        avg_inputs = (
            sum(input_counts.values()) / max(1, len(input_counts))
            if input_counts
            else 2.0
        )

        if self.kits_affordable_at_reserve < 45.0:
            init_mult = self.costly_prior_markup
        elif self.single_good_market:
            init_mult = self.single_good_prior_markup
        else:
            init_mult = self.prior_markup

        for g in goods_config.finished_goods:
            cost = max(self.good_cost.get(g, 1.0), 1e-6)
            res_cost = self.good_reserve_cost.get(g, 0.0)
            curr_inv = max(0, int(state.inventory.get(g, 0)))

            if g not in self.ref_price:
                if self.nominal_reserve_regime:
                    base_prior = max(cost * 5.0, self.starting_capital * 0.025)
                    scale = input_counts.get(g, 2) / max(1.0, avg_inputs)
                    self.ref_price[g] = max(1.0, base_prior * (0.65 + 0.55 * scale))
                else:
                    self.ref_price[g] = max(1.0, cost * init_mult)
                endow = self.init_endowment.get(g, 0)
                self.sales_rate[g] = max(1.0, endow / 4.0) if endow >= 10 else 1.0
                self.floor_mult[g] = self.min_markup
                self.zero_streak[g] = 0
                self.total_sold[g] = 0

            sold = sorted(own_sold_prices.get(g, []))
            n_sold = len(sold)
            self.total_sold[g] = self.total_sold.get(g, 0) + n_sold

            ladder = self.last_ladder.get(g, [])
            actual_listed = min(len(ladder), n_sold + curr_inv)
            tot_sold += n_sold
            tot_listed += actual_listed

            # Fast censored-demand ramp on sellouts so we scale rapidly into demand waves.
            obs_demand = (
                max(n_sold * 1.45, self.sales_rate[g] * 1.35)
                if (actual_listed > 0 and n_sold >= actual_listed)
                else float(n_sold)
            )
            self.sales_rate[g] = 0.65 * self.sales_rate[g] + 0.35 * obs_demand

            fm = self.floor_mult.get(g, self.min_markup)
            if actual_listed > 0:
                if n_sold > 0:
                    self.zero_streak[g] = 0
                    fm = min(self.min_markup, fm + 0.08)
                else:
                    self.zero_streak[g] = self.zero_streak.get(g, 0) + 1
                    if (
                        self.zero_streak[g] >= self.floor_patience
                        and self.ref_price[g] <= cost * fm * 1.03
                    ):
                        fm = max(self.floor_min, fm * self.floor_relax)
            self.floor_mult[g] = fm

            if self.zero_streak.get(g, 0) >= self.floor_patience + 2:
                floor = max(0.05, cost * fm)
            else:
                floor = max(0.05, cost * fm, res_cost * 1.05)

            if actual_listed > 0:
                up = self.step_up * (2.0 if early else 1.0)
                down = self.step_down * (1.5 if early else 1.0)
                ref = self.ref_price[g]
                if n_sold >= actual_listed:
                    mono_bonus = 0.04 if self.rival_stock.get(g, 0) == 0 else 0.0
                    ref *= 1.0 + 2.0 * up + mono_bonus
                elif n_sold > 0 and sold[-1] >= ref * self.up_thresh:
                    ref *= 1.0 + up
                elif n_sold > 0:
                    ref *= 1.0 - 0.30 * down
                else:
                    if self.rival_sold.get(g, 0) > 0:
                        ref *= 1.0 - min(0.32, 1.15 * down)
                    else:
                        ref *= 1.0 - down

                r_prices = rival_sold_prices.get(g)
                if n_sold == 0 and r_prices:
                    top_rival = max(r_prices)
                    if self.total_sold[g] == 0:
                        ref = max(floor, top_rival * 1.05)
                    else:
                        ref = min(ref, top_rival * 1.02)

                self.ref_price[g] = max(ref, floor)
            else:
                r_prices = rival_sold_prices.get(g)
                if self.total_sold[g] == 0 and r_prices:
                    self.ref_price[g] = max(floor, max(r_prices) * 1.05)
                else:
                    self.ref_price[g] = max(self.ref_price[g], floor)

        self.tot_sold_last_turn = tot_sold
        self.tot_listed_last_turn = tot_listed
        self.overall_sell_through = (
            tot_sold / float(tot_listed) if tot_listed > 0 else 0.0
        )

    # ------------------------------------------------------------------
    # Unified target stock, ranking, and build plan
    # ------------------------------------------------------------------

    def _plan_targets_and_needs(
        self, state: Observation, goods_config: GoodsConfig
    ) -> None:
        total_lot_units = sum(max(0, int(lot.quantity)) for lot in state.auction_lots)
        self.supply_per_team = total_lot_units / float(max(1, self.active_team_count))
        self.is_hyper_crowded = self.active_team_count >= 16 and self.supply_per_team < 6.0

        # Unlock full compounding once the market proves strong demand or realized profit
        # (e.g. default_extended 54-team wave demand), while protecting capital in tiny-demand
        # hyper-crowded rooms (e.g. 54-team default where 47/54 teams finish at a loss).
        if not self.proven_hot_market:
            if (
                state.capital >= self.starting_capital * 1.015
                or (
                    self.tot_listed_last_turn >= 2
                    and self.tot_sold_last_turn >= 2
                    and self.overall_sell_through >= 0.65
                )
                or (self.turn >= 4 and self.supply_per_team >= 6.0)
            ):
                self.proven_hot_market = True
        elif (
            self.is_hyper_crowded
            and state.capital < self.starting_capital * 0.99
            and self.overall_sell_through < 0.25
        ):
            self.proven_hot_market = False

        probe_only = self.is_hyper_crowded and not self.proven_hot_market

        stalled_early = (
            self.turn >= 2
            and self.tot_listed_last_turn >= 3
            and self.overall_sell_through < 0.30
        )
        single_endowed = (
            self.single_good_market
            and sum(self.init_endowment.values()) >= 10
        )
        # Never run a blind aggressive opening in hyper-crowded tiny-supply rooms.
        self.open_active = (
            self.turn <= self.open_turns
            and self.kits_affordable_at_reserve >= 45.0
            and not self.nominal_reserve_regime
            and not stalled_early
            and not single_endowed
            and not self.single_good_market
            and not self.is_hyper_crowded
        )

        base_stock = 1 if probe_only else (self.open_stock if self.open_active else self.min_stock)
        target: dict[str, int] = {}
        for g in goods_config.finished_goods:
            curr_inv = max(0, int(state.inventory.get(g, 0)))
            if self.zero_streak.get(g, 0) >= 3 and curr_inv >= 2:
                target[g] = curr_inv
            elif probe_only:
                target[g] = 2
            else:
                t = int(round(self.sales_rate.get(g, 1.0) * self.cover_turns)) + base_stock
                target[g] = max(1, min(self.max_stock, t))

        def good_priority(g: str) -> float:
            rec = goods_config.recipes[self.recipe_for[g]]
            n_in = max(1, sum(int(q) for q in rec.inputs.values()))
            margin_per_in = (self.ref_price[g] - self.good_cost.get(g, 0.0)) / n_in
            if self.active_team_count >= 10:
                holders = self.rival_holders.get(g, 0)
                if holders == 0:
                    margin_per_in *= 1.12
                else:
                    margin_per_in /= 1.0 + 0.025 * holders
            return margin_per_in

        ranked = sorted(
            (
                g
                for g in goods_config.finished_goods
                if g in self.recipe_for
                and self.ref_price.get(g, 0.0) > self.good_cost.get(g, 0.0)
            ),
            key=good_priority,
            reverse=True,
        )

        need: dict[str, int] = {}
        rem_cap = self.estimated_capacity_per_turn
        for idx, g in enumerate(ranked):
            if probe_only and idx >= 2:
                # While probing a hyper-crowded room, focus strictly on top-2 highest-margin goods.
                need[g] = 0
                continue
            curr_inv = max(0, int(state.inventory.get(g, 0)))
            want_g = max(0, target[g] - curr_inv)
            if rem_cap is not None:
                want_g = min(want_g, max(0, rem_cap))
                rem_cap -= want_g
            need[g] = want_g

        self.target_stock = target
        self.ranked_goods = ranked
        self.need_build = need

    def active_floor(self, good: str) -> float:
        """Return the current floor price for `good`."""
        cost = self.good_cost.get(good, 0.0)
        res_cost = self.good_reserve_cost.get(good, 0.0)
        fm = self.floor_mult.get(good, self.min_markup)
        if self.zero_streak.get(good, 0) >= self.floor_patience + 2:
            return max(0.05, cost * fm)
        return max(0.05, cost * fm, res_cost * 1.05)
