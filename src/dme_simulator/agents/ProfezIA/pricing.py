"""Phase 3: Finished-goods pricing & sales strategy (Owner: Persona 2).

Keep all pricing logic and any pricing-specific internal state inside this
module (e.g. attributes on `self`) so changes never conflict with other files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite

from dme_simulator.agents import Listing, Observation
from dme_simulator.config import GoodsConfig

from .state import MarketContext


@dataclass
class _GoodPricingState:
    """Internal per-good state for dynamic WTP discovery and inventory pacing."""

    init_cost: float = 10.0
    input_units: int = 1
    init_endowment: int = 0
    anchor_price: float = 100.0
    price_ceiling: float = 0.0
    max_sold_price: float = 0.0
    ema_sold_price: float = 0.0
    ema_production: float = 1.0
    ema_sales: float = 1.0
    prev_inventory: int = 0
    prev_competitor_stock: int = 0
    prev_listings: list[tuple[int, float, str]] = field(default_factory=list)
    zero_sale_streak: int = 0
    sold_out_streak: int = 0
    has_ever_sold: bool = False


class PricingStrategy:
    """Adaptive multi-tier pricing with staircase WTP discovery and ceiling tracking."""

    def __init__(self) -> None:
        self._initialized: bool = False
        self._starting_capital: float = 10000.0
        self._est_total_turns: int = 40
        self._init_reserve: dict[str, float] = {}
        self._goods: dict[str, _GoodPricingState] = {}
        self._good_to_recipe: dict[str, str] = {}

    def decide_listings(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> list[Listing]:
        """Return `Listing` offers for `goods_config.finished_goods`."""
        if not self._initialized or state.turn == 1:
            self._init_turn_one(state, goods_config)
        else:
            self._update_from_turn_results(state, goods_config, ctx)

        if state.turn >= self._est_total_turns:
            self._est_total_turns = state.turn + 2

        remaining_turns = max(1, self._est_total_turns - state.turn + 1)
        min_prod, exp_prod, max_prod = self._estimate_current_production(
            state, goods_config, ctx
        )

        listings: list[Listing] = []
        for good in goods_config.finished_goods:
            gp = self._goods.get(good)
            if gp is None:
                continue

            curr_inv = max(0, int(state.inventory.get(good, 0)))
            expected_stock = curr_inv + exp_prod.get(good, 0)
            optimistic_stock = curr_inv + max_prod.get(good, 0)
            comp_stock = self._competitor_finished_stock(state, ctx, good)

            unit_cost = self._current_unit_cost(good, state, goods_config, ctx, gp)
            cost_floor = self._compute_cost_floor(
                unit_cost, gp, curr_inv, remaining_turns
            )

            base_anchor = max(cost_floor, gp.anchor_price)
            if comp_stock == 0 and gp.has_ever_sold and remaining_turns > 3 and curr_inv == 0:
                base_anchor = max(base_anchor, gp.ema_sold_price * 1.025)

            good_tiers = self._build_good_staircase(
                good=good,
                gp=gp,
                curr_inv=curr_inv,
                expected_stock=expected_stock,
                optimistic_stock=optimistic_stock,
                comp_stock=comp_stock,
                base_anchor=base_anchor,
                cost_floor=cost_floor,
                remaining_turns=remaining_turns,
                turn=state.turn,
            )

            gp.prev_inventory = curr_inv
            gp.prev_competitor_stock = comp_stock
            gp.prev_listings = good_tiers

            for qty, price, _role in good_tiers:
                if qty > 0 and isfinite(price) and price > 0:
                    listings.append(Listing(good, int(qty), round(float(price), 2)))

        return listings

    # ------------------------------------------------------------------
    # Initialization & structural WTP prior
    # ------------------------------------------------------------------

    def _init_turn_one(
        self, state: Observation, goods_config: GoodsConfig
    ) -> None:
        self._initialized = True
        self._starting_capital = max(1.0, float(state.capital))
        self._init_reserve = {
            lot.good: max(0.01, float(lot.reserve_price))
            for lot in state.auction_lots
        }
        self._good_to_recipe.clear()
        for rname, recipe in goods_config.recipes.items():
            self._good_to_recipe.setdefault(recipe.output_good, rname)

        total_initial_lot_units = sum(max(0, lot.quantity) for lot in state.auction_lots)
        num_finished = len(goods_config.finished_goods)
        if num_finished <= 2 and self._starting_capital <= 8000.0:
            self._est_total_turns = 20
        elif self._starting_capital >= 30000.0 or total_initial_lot_units >= 500:
            self._est_total_turns = 30
        else:
            self._est_total_turns = 40

        init_costs: dict[str, float] = {}
        input_counts: dict[str, int] = {}
        for good in goods_config.finished_goods:
            rname = self._good_to_recipe.get(good)
            if rname and rname in goods_config.recipes:
                recipe = goods_config.recipes[rname]
                c = sum(
                    self._init_reserve.get(mat, 5.0) * qty
                    for mat, qty in recipe.inputs.items()
                )
                u = sum(int(qty) for qty in recipe.inputs.values())
            else:
                c = 25.0
                u = 2
            init_costs[good] = max(0.5, c)
            input_counts[good] = max(1, u)

        c_min = min(init_costs.values()) if init_costs else 1.0
        c_max = max(init_costs.values()) if init_costs else 1.0
        u_min = min(input_counts.values()) if input_counts else 1
        u_max = max(input_counts.values()) if input_counts else 1

        self._goods.clear()
        for good in goods_config.finished_goods:
            icost = init_costs[good]
            iunits = input_counts[good]

            if num_finished > 1 and (c_max - c_min > 1e-6 or u_max > u_min):
                r_cost = (icost - c_min) / max(1e-6, c_max - c_min)
                r_units = (iunits - u_min) / max(1e-6, float(u_max - u_min))
                r_tier = 0.72 * r_cost + 0.28 * r_units
                mult = 5.3 + 10.6 * (r_tier ** 1.2)
                anchor = icost * mult
            else:
                if icost < self._starting_capital * 0.002:
                    anchor = max(icost * 5.0, self._starting_capital * 0.0285)
                else:
                    anchor = icost * 5.0

            init_inv = max(0, int(state.inventory.get(good, 0)))
            self._goods[good] = _GoodPricingState(
                init_cost=icost,
                input_units=iunits,
                init_endowment=init_inv,
                anchor_price=max(icost * 1.25, anchor),
                price_ceiling=0.0,
                ema_production=1.0,
                ema_sales=max(1.0, init_inv / 6.0) if init_inv > 0 else 1.0,
                prev_inventory=init_inv,
            )

    # ------------------------------------------------------------------
    # Turn-over-turn learning from realized sales & ceiling detection
    # ------------------------------------------------------------------

    def _update_from_turn_results(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> None:
        our_txs_by_good: dict[str, list[tuple[int, float]]] = {
            good: [] for good in goods_config.finished_goods
        }
        comp_max_price: dict[str, float] = {}

        for tx in ctx.new_sales_txs:
            if tx.quantity <= 0 or tx.price <= 0:
                continue
            if ctx.team_id is None or tx.seller == ctx.team_id:
                our_txs_by_good.setdefault(tx.good, []).append(
                    (int(tx.quantity), float(tx.price))
                )
            else:
                prev_max = comp_max_price.get(tx.good, 0.0)
                if tx.price > prev_max:
                    comp_max_price[tx.good] = float(tx.price)

        prev_remaining_turns = max(1, self._est_total_turns - (state.turn - 1) + 1)
        stock_horizon = min(float(prev_remaining_turns), 11.0)
        num_goods = max(1, len(goods_config.finished_goods))

        for good in goods_config.finished_goods:
            gp = self._goods.get(good)
            if gp is None:
                continue

            tx_list = our_txs_by_good.get(good, [])
            sold_qty = sum(q for q, _ in tx_list)
            sold_rev = sum(q * p for q, p in tx_list)
            curr_inv = max(0, int(state.inventory.get(good, 0)))

            # Exact Phase-3 available stock on turn t-1 by inventory conservation:
            actual_available = curr_inv + sold_qty
            actual_produced = max(0, actual_available - gp.prev_inventory)

            gp.ema_production = 0.35 * actual_produced + 0.65 * gp.ema_production
            gp.ema_sales = 0.35 * sold_qty + 0.65 * gp.ema_sales

            if good in comp_max_price:
                gp.max_sold_price = max(gp.max_sold_price, comp_max_price[good])

            if actual_available <= 0:
                continue

            # Replay how many units were actually offered at each price tier on turn t-1.
            rem = actual_available
            active_offered = 0
            offered_tiers: list[tuple[int, float, str]] = []
            for req_qty, tier_price, role in gp.prev_listings:
                if rem <= 0:
                    break
                off = min(rem, req_qty)
                rem -= off
                if off > 0:
                    offered_tiers.append((off, tier_price, role))
                    if role != "reserve":
                        active_offered += off
            active_offered = max(1, active_offered)

            is_endowment_good = (
                num_goods == 1
                and gp.init_endowment >= 10
                and gp.ema_production <= 1.5
            )
            if is_endowment_good:
                target_velocity = max(
                    1.0,
                    (gp.prev_inventory / stock_horizon) + float(actual_produced),
                )
            else:
                # Active production regime: must clear new production plus drain backlog.
                target_velocity = max(
                    1.0,
                    float(actual_produced) + (gp.prev_inventory / min(float(prev_remaining_turns), 3.0)),
                )

            curr_comp_stock = self._competitor_finished_stock(state, ctx, good)
            competitor_dumped = (
                gp.prev_competitor_stock >= 12
                and curr_comp_stock < gp.prev_competitor_stock * 0.65
            )

            if sold_qty > 0:
                gp.has_ever_sold = True
                gp.zero_sale_streak = 0
                avg_tx_price = sold_rev / sold_qty
                max_tx_price = max(p for _, p in tx_list)
                min_tx_price = min(p for _, p in tx_list)

                gp.max_sold_price = max(gp.max_sold_price * 0.995, max_tx_price)
                if gp.ema_sold_price <= 0:
                    gp.ema_sold_price = avg_tx_price
                else:
                    gp.ema_sold_price = (
                        0.45 * avg_tx_price + 0.55 * gp.ema_sold_price
                    )

                # Gently relax any existing ceiling over time so it never traps prices.
                if gp.price_ceiling > 0:
                    if max_tx_price >= gp.price_ceiling * 0.97:
                        gp.price_ceiling = max_tx_price * 1.08
                    else:
                        gp.price_ceiling *= 1.015

                # Only infer a tight upper WTP ceiling when a meaningful sample of customers
                # (sold_qty >= 3) bought lower tiers while rejecting the next tier above.
                if not competitor_dumped and sold_qty >= 3:
                    failed_above = [
                        tp
                        for off_q, tp, role in offered_tiers
                        if off_q > 0 and role != "reserve" and tp > max_tx_price * 1.005
                    ]
                    if failed_above:
                        lowest_failed = min(failed_above)
                        if gp.price_ceiling <= 0:
                            gp.price_ceiling = lowest_failed
                        else:
                            gp.price_ceiling = min(
                                gp.price_ceiling,
                                0.35 * gp.price_ceiling + 0.65 * lowest_failed,
                            )

                sold_all_available = sold_qty >= actual_available
                sold_all_active = sold_qty >= active_offered
                met_velocity = sold_qty >= max(1, int(round(0.85 * target_velocity)))

                if sold_all_available:
                    gp.sold_out_streak += 1
                    step_up = 1.07 if gp.sold_out_streak == 1 else 1.095
                    new_anchor = max(gp.anchor_price * 1.045, max_tx_price * step_up)
                    if gp.price_ceiling > 0:
                        gp.price_ceiling = max(gp.price_ceiling, new_anchor * 1.06)
                    gp.anchor_price = new_anchor
                elif sold_all_active or (
                    met_velocity and max_tx_price >= gp.anchor_price * 0.99
                ):
                    gp.sold_out_streak += 1
                    gp.anchor_price = max(
                        gp.anchor_price * 1.02,
                        0.40 * gp.anchor_price + 0.60 * max_tx_price * 1.025,
                    )
                elif not is_endowment_good and curr_inv >= 5 and sold_qty < 0.70 * target_velocity:
                    # Heavy backlog accumulating: ease anchor down to clear excess shelf stock.
                    gp.sold_out_streak = 0
                    ref_price = 0.60 * avg_tx_price + 0.40 * min_tx_price
                    gp.anchor_price = (0.50 * gp.anchor_price + 0.50 * ref_price) * 0.94
                elif sold_qty < 0.55 * target_velocity and not competitor_dumped:
                    gp.sold_out_streak = 0
                    gp.anchor_price = (
                        0.60 * gp.anchor_price + 0.40 * avg_tx_price
                    ) * 0.96
                else:
                    gp.sold_out_streak = 0
                    target_ref = max_tx_price if sold_qty <= 2 else (0.50 * avg_tx_price + 0.50 * max_tx_price)
                    if gp.price_ceiling > 0:
                        target_ref = min(target_ref, gp.price_ceiling * 0.98)
                    gp.anchor_price = 0.50 * gp.anchor_price + 0.50 * target_ref
            else:
                # 0 units sold despite having positive stock in Phase 3.
                gp.sold_out_streak = 0
                if competitor_dumped:
                    gp.anchor_price *= 0.99
                else:
                    gp.zero_sale_streak += 1
                    if gp.has_ever_sold and gp.ema_sold_price > 0:
                        decay = 0.92 if gp.zero_sale_streak == 1 else 0.85
                        gp.anchor_price = min(
                            gp.anchor_price * decay,
                            0.50 * gp.anchor_price + 0.50 * gp.ema_sold_price,
                        )
                    else:
                        if gp.zero_sale_streak == 1:
                            gp.anchor_price *= 0.84
                        elif gp.zero_sale_streak == 2:
                            gp.anchor_price *= 0.76
                        else:
                            gp.anchor_price *= 0.70

    # ------------------------------------------------------------------
    # Staircase construction
    # ------------------------------------------------------------------

    def _build_good_staircase(
        self,
        good: str,
        gp: _GoodPricingState,
        curr_inv: int,
        expected_stock: int,
        optimistic_stock: int,
        comp_stock: int,
        base_anchor: float,
        cost_floor: float,
        remaining_turns: int,
        turn: int,
    ) -> list[tuple[int, float, str]]:
        """Construct ascending price staircase `(qty, price, role)` for `good`."""
        stock_to_plan = max(1, expected_stock)
        extra_buffer = max(2, optimistic_stock - stock_to_plan + 6)
        num_goods = max(1, len(self._goods))

        endgame_discount = 1.0
        if remaining_turns == 4 and curr_inv >= 3 and num_goods > 1:
            endgame_discount = 0.93
        elif remaining_turns == 3:
            if stock_to_plan > ceil(gp.ema_sales * 1.6):
                endgame_discount = 0.88
            elif curr_inv >= 2 and num_goods > 1:
                endgame_discount = 0.91
        elif remaining_turns == 2:
            if stock_to_plan > 1:
                endgame_discount = 0.80 if num_goods > 1 else 0.85
            elif num_goods > 1:
                endgame_discount = 0.88
        elif remaining_turns == 1:
            if num_goods > 1:
                endgame_discount = 0.68 if stock_to_plan > 1 else 0.75
            else:
                endgame_discount = 0.80 if stock_to_plan > 2 else 0.88

        anchor = max(cost_floor, base_anchor * endgame_discount)
        if remaining_turns == 1 and gp.ema_sold_price > 0 and num_goods > 1:
            anchor = max(cost_floor, min(anchor, gp.ema_sold_price * 0.78))

        # Respect empirical price ceiling if one has been discovered.
        ceiling = gp.price_ceiling if (gp.price_ceiling > cost_floor and remaining_turns > 1) else 0.0
        if ceiling > 0 and anchor > ceiling * 0.97:
            anchor = max(cost_floor, ceiling * 0.96)

        # Case 1: Only 1 expected unit available in Phase 3.
        if stock_to_plan == 1:
            if gp.zero_sale_streak >= 1:
                first_price = max(cost_floor, anchor * 0.95)
            else:
                first_price = anchor
            probe_price = max(first_price * 1.04, anchor * 1.06)
            if ceiling > 0:
                probe_price = min(probe_price, ceiling * 1.01)
            return [
                (1, first_price, "core"),
                (extra_buffer, max(first_price, probe_price), "probe"),
            ]

        # Case 2: Multiple units available.
        is_endowment_good = (
            num_goods == 1
            and gp.init_endowment >= 10
            and gp.ema_production <= 1.5
        )
        est_flow = max(0, expected_stock - curr_inv)

        if is_endowment_good and remaining_turns > 3:
            stock_horizon = min(float(remaining_turns), 10.0)
            if turn == 1 and comp_stock >= 15 and curr_inv >= 10:
                active_qty = max(3, ceil(curr_inv / 6.0) + est_flow)
            else:
                target_active = ceil(
                    curr_inv / stock_horizon + max(float(est_flow), gp.ema_production) + 0.5
                )
                active_qty = min(stock_to_plan, max(2, target_active))
        else:
            # In active production mode, offer 100% of expected stock in active tiers
            # so produced units never sit blocked behind a synthetic holdback price!
            active_qty = stock_to_plan

        holdback_qty = max(0, stock_to_plan - active_qty)

        max_steps = 10
        chunk = max(1, ceil(active_qty / max_steps))
        tiers: list[tuple[int, float, str]] = []

        if ceiling > 0:
            # Ceiling is known: pack 85%+ of steps strictly below ceiling, with 1 top probe at ceiling.
            if curr_inv >= 4 and not is_endowment_good:
                step_mults = [0.88, 0.93, 0.96, 0.98, 1.00, 1.02]
            else:
                step_mults = [0.95, 0.97, 0.99, 1.00, 1.015, 1.03]
        elif not gp.has_ever_sold and turn == 1:
            if num_goods == 1 and gp.init_endowment == 0:
                # Single-good production auction (e.g. tutorial_auction): tight band around 5x cost.
                step_mults = [0.95, 0.97, 0.99, 1.01, 1.03, 1.06]
            else:
                step_mults = [0.93, 0.98, 1.03, 1.08, 1.14, 1.22]
        elif gp.zero_sale_streak >= 1 or (curr_inv >= 5 and not is_endowment_good):
            step_mults = [0.88, 0.93, 0.97, 1.01, 1.05]
        else:
            step_mults = [0.96, 0.99, 1.02, 1.05, 1.09, 1.14]

        allocated = 0
        step_idx = 0
        while allocated < active_qty:
            qty = min(chunk, active_qty - allocated)
            frac = step_idx / max(1, len(step_mults) - 1)
            idx = min(step_idx, len(step_mults) - 1)
            mult = step_mults[idx] if step_idx < len(step_mults) else (
                step_mults[-1] + 0.025 * (step_idx - len(step_mults) + 1)
            )
            step_price = max(cost_floor, anchor * mult)
            if ceiling > 0 and frac < 0.85:
                step_price = max(cost_floor, min(step_price, ceiling * 0.985))
            elif ceiling > 0:
                step_price = max(cost_floor, min(step_price, ceiling * 1.015))
            role = "core" if frac <= 0.70 else "upsell"
            tiers.append((qty, step_price, role))
            allocated += qty
            step_idx += 1

        top_active_price = tiers[-1][1] if tiers else anchor
        if holdback_qty > 0:
            reserve_price = max(top_active_price * 1.05, anchor * 1.14)
            tiers.append((holdback_qty, reserve_price, "reserve"))
            top_active_price = reserve_price

        if ceiling > 0:
            catchall_price = max(cost_floor, min(top_active_price * 1.01, ceiling * 0.99))
        else:
            catchall_price = max(top_active_price * 1.03, anchor * 1.06)
        tiers.append((extra_buffer, catchall_price, "probe"))
        return tiers

    # ------------------------------------------------------------------
    # Production & cost helpers
    # ------------------------------------------------------------------

    def _estimate_current_production(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        """Simulate `ctx.planned_assembly` under 0%, expected, and 100% bid fill rates."""
        lot_caps = {lot.good: max(0, int(lot.quantity)) for lot in state.auction_lots}
        bid_map: dict[str, int] = {}
        for bid in ctx.planned_bids:
            if bid.quantity > 0:
                cap = lot_caps.get(bid.good, bid.quantity)
                bid_map[bid.good] = min(cap, bid_map.get(bid.good, 0) + int(bid.quantity))

        fill_map = getattr(ctx, "auction_fill_rate", {})
        inv_min = {raw: max(0, int(state.inventory.get(raw, 0))) for raw in goods_config.raw_materials}
        inv_exp = {
            raw: inv_min[raw] + int(round(fill_map.get(raw, 0.75) * bid_map.get(raw, 0)))
            for raw in goods_config.raw_materials
        }
        inv_max = {
            raw: inv_min[raw] + bid_map.get(raw, 0)
            for raw in goods_config.raw_materials
        }

        cap = getattr(ctx, "estimated_capacity_per_turn", None)
        return (
            self._simulate_assembly(inv_min, goods_config, ctx, cap),
            self._simulate_assembly(inv_exp, goods_config, ctx, cap),
            self._simulate_assembly(inv_max, goods_config, ctx, cap),
        )

    @staticmethod
    def _simulate_assembly(
        raw_inv: dict[str, int],
        goods_config: GoodsConfig,
        ctx: MarketContext,
        capacity_limit: int | None = None,
    ) -> dict[str, int]:
        produced: dict[str, int] = {good: 0 for good in goods_config.finished_goods}
        work_inv = dict(raw_inv)
        rem_cap = capacity_limit
        for order in ctx.planned_assembly:
            if order.quantity <= 0 or order.recipe_name not in goods_config.recipes:
                continue
            if rem_cap is not None and rem_cap <= 0:
                break
            recipe = goods_config.recipes[order.recipe_name]
            if not recipe.inputs:
                continue
            buildable = min(
                work_inv.get(mat, 0) // max(1, int(amt))
                for mat, amt in recipe.inputs.items()
            )
            qty = min(int(order.quantity), max(0, buildable))
            if rem_cap is not None:
                qty = min(qty, rem_cap)
            if qty <= 0:
                continue
            for mat, amt in recipe.inputs.items():
                work_inv[mat] = work_inv.get(mat, 0) - int(amt) * qty
            produced[recipe.output_good] = produced.get(recipe.output_good, 0) + qty
            if rem_cap is not None:
                rem_cap -= qty
        return produced

    def _current_unit_cost(
        self,
        good: str,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
        gp: _GoodPricingState,
    ) -> float:
        rname = self._good_to_recipe.get(good)
        if not rname or rname not in goods_config.recipes:
            return gp.init_cost

        ctx_cost = ctx.estimate_recipe_cost(rname, goods_config)
        recipe = goods_config.recipes[rname]
        lot_prices = {lot.good: lot.reserve_price for lot in state.auction_lots}
        reserve_cost = sum(
            lot_prices.get(mat, ctx.avg_material_cost.get(mat, 1.0)) * qty
            for mat, qty in recipe.inputs.items()
        )
        return max(0.5, ctx_cost, reserve_cost, gp.init_cost * 0.75)

    @staticmethod
    def _compute_cost_floor(
        unit_cost: float,
        gp: _GoodPricingState,
        curr_inv: int,
        remaining_turns: int,
    ) -> float:
        if remaining_turns <= 1 and curr_inv > 0:
            return max(1.0, unit_cost * 0.65)
        if remaining_turns == 2 and curr_inv >= 2:
            return max(1.0, unit_cost * 0.85)
        return max(1.0, unit_cost * 1.12)

    @staticmethod
    def _competitor_finished_stock(
        state: Observation, ctx: MarketContext, good: str
    ) -> int:
        total = 0
        for tid, inv in state.public_inventories.items():
            if ctx.team_id is not None and tid == ctx.team_id:
                continue
            total += max(0, int(inv.get(good, 0)))
        if ctx.team_id is None:
            total = max(0, total - max(0, int(state.inventory.get(good, 0))))
        return total
