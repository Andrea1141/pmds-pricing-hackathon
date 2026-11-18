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


_HORIZON_CHECKPOINTS = (20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 240)


@dataclass
class _GoodPricingState:
    """Internal per-good state for dynamic WTP discovery and inventory pacing."""

    init_cost: float = 10.0
    eff_init_cost: float = 10.0
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
    """Adaptive multi-tier pricing with staircase WTP discovery, blitz monopoly extraction, and crowding clearance."""

    def __init__(self) -> None:
        self._initialized: bool = False
        self._starting_capital: float = 10000.0
        self._est_total_turns: int = 40
        self._init_reserve: dict[str, float] = {}
        self._goods: dict[str, _GoodPricingState] = {}
        self._good_to_recipes: dict[str, list[str]] = {}

    def decide_listings(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> list[Listing]:
        """Return `Listing` offers for `goods_config.finished_goods`."""
        if not self._initialized or state.turn == 1:
            self._init_turn_one(state, goods_config, ctx)
        else:
            self._update_from_turn_results(state, goods_config, ctx)

        # Advance horizon along standard tournament checkpoints and rebound anchors
        # after crossing a checkpoint so 60/80/240-turn matches still clear at the end.
        if state.turn > self._est_total_turns:
            next_cp = next((cp for cp in _HORIZON_CHECKPOINTS if cp >= state.turn), None)
            self._est_total_turns = next_cp if next_cp is not None else state.turn + 40
            for gp in self._goods.values():
                if gp.has_ever_sold:
                    gp.anchor_price = max(
                        gp.anchor_price * 1.18,
                        gp.ema_sold_price * 1.12,
                        gp.max_sold_price * 0.94,
                    )
                    if gp.price_ceiling > 0.0:
                        gp.price_ceiling = max(gp.price_ceiling, gp.anchor_price * 1.08)

        remaining_turns = max(1, self._est_total_turns - state.turn + 1)
        min_prod, exp_prod, max_prod = self._estimate_current_production(
            state, goods_config, ctx
        )
        demand_heat = float(getattr(ctx, "market_demand_heat", 1.0))
        is_crowded = bool(
            getattr(ctx, "is_crowded_tournament", len(state.public_capital) >= 10)
        )
        is_expansion = bool(getattr(ctx, "is_high_volume_expansion", False))
        is_blitz = bool(
            getattr(ctx, "is_blitz_window", is_crowded and state.turn <= 6)
        )
        sellers_map = getattr(ctx, "competitor_sellers_by_good", {})
        crowding_map = getattr(ctx, "crowding_ratio_by_good", {})
        mkt_max_map = getattr(ctx, "market_max_sale_price", {})

        listings: list[Listing] = []
        for good in goods_config.finished_goods:
            gp = self._goods.get(good)
            if gp is None:
                continue

            curr_inv = max(0, int(state.inventory.get(good, 0)))
            expected_stock = curr_inv + exp_prod.get(good, 0)
            optimistic_stock = curr_inv + max_prod.get(good, 0)
            comp_stock = self._competitor_finished_stock(state, ctx, good)
            comp_sellers = int(
                sellers_map.get(good, 0 if comp_stock == 0 else 2)
            )
            crowding = float(crowding_map.get(good, 0.0))

            unit_cost = self._current_unit_cost(good, state, goods_config, ctx, gp)
            cost_floor = self._compute_cost_floor(
                unit_cost, gp, curr_inv, remaining_turns, is_crowded
            )

            base_anchor = max(cost_floor, gp.anchor_price)

            # Objective 1: Monopoly & near-monopoly extraction during Blitz and open niches.
            if is_crowded:
                if (
                    state.turn >= 2
                    and comp_sellers <= 1
                    and gp.zero_sale_streak == 0
                    and curr_inv <= 3
                    and remaining_turns > 4
                ):
                    if comp_sellers == 0:
                        mono_mult = 1.14 if is_blitz else 1.10
                    else:
                        mono_mult = 1.09 if is_blitz else 1.06
                    base_anchor = max(
                        base_anchor * mono_mult,
                        gp.ema_sold_price * mono_mult if gp.ema_sold_price > 0 else 0.0,
                    )
                    mkt_max = float(mkt_max_map.get(good, 0.0))
                    if mkt_max > base_anchor:
                        base_anchor = 0.65 * base_anchor + 0.35 * mkt_max
            else:
                if comp_stock == 0 and gp.has_ever_sold and remaining_turns > 3 and curr_inv <= 1:
                    base_anchor = max(base_anchor, gp.ema_sold_price * 1.03)

            if demand_heat > 1.25 and gp.zero_sale_streak == 0 and curr_inv <= 2:
                base_anchor *= min(1.06, 1.0 + 0.08 * (demand_heat - 1.20))

            good_tiers = self._build_good_staircase(
                good=good,
                gp=gp,
                curr_inv=curr_inv,
                expected_stock=expected_stock,
                optimistic_stock=optimistic_stock,
                comp_stock=comp_stock,
                comp_sellers=comp_sellers,
                crowding=crowding,
                is_crowded=is_crowded,
                is_expansion=is_expansion,
                is_blitz=is_blitz,
                base_anchor=base_anchor,
                cost_floor=cost_floor,
                remaining_turns=remaining_turns,
                turn=state.turn,
                demand_heat=demand_heat,
            )

            gp.prev_inventory = curr_inv
            gp.prev_competitor_stock = comp_stock
            gp.prev_listings = good_tiers

            for qty, price, _role in good_tiers:
                if qty > 0 and isfinite(price) and price > 0:
                    decimals = 4 if price < 5.0 else 2
                    rounded = max(0.01, round(float(price), decimals))
                    listings.append(Listing(good, int(qty), rounded))

        return listings

    # ------------------------------------------------------------------
    # Initialization & structural WTP prior (handles multi-recipe, zero-reserve & 38-team rooms)
    # ------------------------------------------------------------------

    def _init_turn_one(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext | None = None,
    ) -> None:
        self._initialized = True
        self._starting_capital = max(1.0, float(state.capital))
        self._init_reserve = {
            lot.good: max(0.0, float(lot.reserve_price))
            for lot in state.auction_lots
        }
        self._good_to_recipes.clear()
        for rname, recipe in goods_config.recipes.items():
            self._good_to_recipes.setdefault(recipe.output_good, []).append(rname)

        total_initial_lot_units = sum(max(0, lot.quantity) for lot in state.auction_lots)
        num_finished = max(1, len(goods_config.finished_goods))
        if num_finished <= 2 and self._starting_capital <= 8000.0:
            self._est_total_turns = 20
        elif self._starting_capital >= 30000.0 or total_initial_lot_units >= 500:
            self._est_total_turns = 30
        else:
            self._est_total_turns = 40

        is_crowded = (
            bool(getattr(ctx, "is_crowded_tournament", False))
            if ctx is not None
            else (len(state.public_capital) >= 10)
        )

        raw_init_costs: dict[str, float] = {}
        input_counts: dict[str, int] = {}
        for good in goods_config.finished_goods:
            rnames = self._good_to_recipes.get(good, [])
            if rnames:
                best_c = float("inf")
                best_u = 1
                for rname in rnames:
                    recipe = goods_config.recipes[rname]
                    c = sum(
                        self._init_reserve.get(mat, 0.0) * qty
                        for mat, qty in recipe.inputs.items()
                    )
                    u = sum(int(qty) for qty in recipe.inputs.values())
                    if c < best_c or (abs(c - best_c) < 1e-6 and u < best_u):
                        best_c = c
                        best_u = u
                raw_init_costs[good] = max(0.0, best_c)
                input_counts[good] = max(1, best_u)
            else:
                raw_init_costs[good] = 25.0
                input_counts[good] = 2

        avg_units = (
            sum(input_counts.values()) / max(1, len(input_counts))
            if input_counts
            else 2.0
        )
        max_raw_cost = max(raw_init_costs.values()) if raw_init_costs else 0.0
        nominal_cost_regime = (
            (max_raw_cost <= 0.05 and self._starting_capital >= 500.0)
            or (
                max_raw_cost <= 3.0
                and self._starting_capital >= 1000.0
                and total_initial_lot_units >= 300
            )
        )

        eff_costs: dict[str, float] = {}
        for good in goods_config.finished_goods:
            rc = raw_init_costs[good]
            iu = input_counts[good]
            if nominal_cost_regime:
                if num_finished == 1:
                    eff_costs[good] = max(rc, self._starting_capital * 0.0052)
                else:
                    eff_costs[good] = max(
                        rc,
                        (self._starting_capital * 0.0045) * (iu / max(1.0, avg_units)),
                    )
            else:
                eff_costs[good] = max(0.01, rc)

        c_min = min(eff_costs.values()) if eff_costs else 1.0
        c_max = max(eff_costs.values()) if eff_costs else 1.0
        u_min = min(input_counts.values()) if input_counts else 1
        u_max = max(input_counts.values()) if input_counts else 1

        self._goods.clear()
        for good in goods_config.finished_goods:
            icost = max(0.01, raw_init_costs[good])
            ecost = eff_costs[good]
            iunits = input_counts[good]

            if num_finished > 1 and (c_max - c_min > 1e-6 or u_max > u_min):
                r_cost = (ecost - c_min) / max(1e-6, c_max - c_min)
                r_units = (iunits - u_min) / max(1e-6, float(u_max - u_min))
                r_tier = 0.72 * r_cost + 0.28 * r_units if not nominal_cost_regime else r_units
                if is_crowded:
                    mult = 5.0 + 5.6 * (r_tier ** 1.15)
                else:
                    mult = 5.3 + 10.6 * (r_tier ** 1.2)
                anchor = ecost * mult
            else:
                if nominal_cost_regime:
                    anchor = max(icost * 5.0, self._starting_capital * 0.0265)
                else:
                    anchor = ecost * 5.0

            init_inv = max(0, int(state.inventory.get(good, 0)))
            self._goods[good] = _GoodPricingState(
                init_cost=icost,
                eff_init_cost=ecost,
                input_units=iunits,
                init_endowment=init_inv,
                anchor_price=max(icost * 1.25, anchor),
                price_ceiling=0.0,
                ema_production=1.0,
                ema_sales=max(1.5, init_inv / 4.5) if init_inv > 0 else 1.0,
                prev_inventory=init_inv,
            )

    # ------------------------------------------------------------------
    # Turn-over-turn learning from realized sales, demand heat & crowding
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

        is_own_sale = getattr(ctx, "is_own_sales_tx", None)
        for tx in ctx.new_sales_txs:
            if tx.quantity <= 0 or tx.price <= 0:
                continue
            own_tx = (
                is_own_sale(tx)
                if is_own_sale is not None
                else (ctx.team_id is not None and tx.seller == ctx.team_id)
            )
            if own_tx:
                our_txs_by_good.setdefault(tx.good, []).append(
                    (int(tx.quantity), float(tx.price))
                )
            else:
                prev_max = comp_max_price.get(tx.good, 0.0)
                if tx.price > prev_max:
                    comp_max_price[tx.good] = float(tx.price)

        prev_remaining_turns = max(1, self._est_total_turns - (state.turn - 1) + 1)
        demand_heat = float(getattr(ctx, "market_demand_heat", 1.0))
        is_crowded = bool(
            getattr(ctx, "is_crowded_tournament", len(state.public_capital) >= 10)
        )
        sellers_map = getattr(ctx, "competitor_sellers_by_good", {})
        crowding_map = getattr(ctx, "crowding_ratio_by_good", {})

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

            inventory_cover = curr_inv / max(0.5, gp.ema_sales)
            target_velocity = max(
                1.0,
                float(actual_produced) + (gp.prev_inventory / min(float(prev_remaining_turns), 4.0)),
            )

            curr_comp_stock = self._competitor_finished_stock(state, ctx, good)
            comp_sellers = int(sellers_map.get(good, 0 if curr_comp_stock == 0 else 2))
            crowding = float(crowding_map.get(good, 0.0))

            competitor_dumped = (
                state.turn <= 3
                and not is_crowded
                and gp.prev_competitor_stock >= 12
                and curr_comp_stock < gp.prev_competitor_stock * 0.55
            )
            high_rival_pressure = curr_comp_stock > max(2, curr_inv) or (
                is_crowded and (crowding >= 0.40 or comp_sellers >= 5)
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

                if gp.price_ceiling > 0:
                    if max_tx_price >= gp.price_ceiling * 0.97:
                        gp.price_ceiling = max_tx_price * 1.08
                    else:
                        gp.price_ceiling *= 1.015

                if not competitor_dumped and demand_heat >= 0.72 and sold_qty >= 3:
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
                    if is_crowded and comp_sellers <= 1:
                        step_up = 1.10 if gp.sold_out_streak == 1 else 1.13
                    else:
                        step_up = 1.075 if gp.sold_out_streak == 1 else 1.10
                    new_anchor = max(gp.anchor_price * 1.05, max_tx_price * step_up)
                    if gp.price_ceiling > 0:
                        gp.price_ceiling = max(gp.price_ceiling, new_anchor * 1.06)
                    gp.anchor_price = new_anchor
                elif sold_all_active or (
                    met_velocity and max_tx_price >= gp.anchor_price * 0.99 and inventory_cover <= 2.0
                ):
                    gp.sold_out_streak += 1
                    gp.anchor_price = max(
                        gp.anchor_price * 1.025,
                        0.40 * gp.anchor_price + 0.60 * max_tx_price * 1.03,
                    )
                elif (
                    is_crowded
                    and (crowding >= 0.40 or (comp_sellers >= 4 and inventory_cover >= 1.35))
                    and curr_inv >= 2
                    and sold_qty < target_velocity
                ):
                    # Objective 3: Fast clearance in crowded 38-team goods before price war deepens.
                    gp.sold_out_streak = 0
                    ref_price = 0.45 * avg_tx_price + 0.55 * min_tx_price
                    decay = 0.89 if crowding >= 0.45 else 0.92
                    gp.anchor_price = min(
                        gp.anchor_price * decay,
                        (0.35 * gp.anchor_price + 0.65 * ref_price) * decay,
                    )
                elif curr_inv >= 3 and (
                    inventory_cover > 2.1 or (high_rival_pressure and sold_qty < target_velocity)
                ):
                    gp.sold_out_streak = 0
                    ref_price = 0.50 * avg_tx_price + 0.50 * min_tx_price
                    decay = 0.92 if high_rival_pressure else 0.95
                    if demand_heat < 0.68 and not is_crowded:
                        decay = max(decay, 0.97)
                    gp.anchor_price = (0.45 * gp.anchor_price + 0.55 * ref_price) * decay
                elif sold_qty < 0.55 * target_velocity and not competitor_dumped:
                    gp.sold_out_streak = 0
                    decay = 0.98 if (demand_heat < 0.68 and not is_crowded) else (0.93 if is_crowded else 0.95)
                    gp.anchor_price = (
                        0.55 * gp.anchor_price + 0.45 * avg_tx_price
                    ) * decay
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
                    gp.anchor_price *= 0.98
                elif not is_crowded and demand_heat < 0.65 and gp.zero_sale_streak == 0 and inventory_cover <= 2.5:
                    gp.zero_sale_streak += 1
                    gp.anchor_price *= 0.96
                else:
                    gp.zero_sale_streak += 1
                    if gp.has_ever_sold and gp.ema_sold_price > 0:
                        decay = 0.90 if gp.zero_sale_streak == 1 else 0.82
                        if is_crowded and (crowding >= 0.40 or curr_inv >= 2):
                            decay -= 0.05
                        elif high_rival_pressure and inventory_cover > 2.0:
                            decay -= 0.04
                        gp.anchor_price = min(
                            gp.anchor_price * decay,
                            0.45 * gp.anchor_price + 0.55 * gp.ema_sold_price,
                        )
                    else:
                        if is_crowded:
                            decay = 0.78 if gp.zero_sale_streak == 1 else (0.70 if gp.zero_sale_streak == 2 else 0.64)
                            gp.anchor_price *= decay
                        else:
                            if gp.zero_sale_streak == 1:
                                gp.anchor_price *= 0.82
                            elif gp.zero_sale_streak == 2:
                                gp.anchor_price *= 0.74
                            else:
                                gp.anchor_price *= 0.66

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
        comp_sellers: int,
        crowding: float,
        is_crowded: bool,
        is_expansion: bool,
        is_blitz: bool,
        base_anchor: float,
        cost_floor: float,
        remaining_turns: int,
        turn: int,
        demand_heat: float,
    ) -> list[tuple[int, float, str]]:
        """Construct ascending price staircase `(qty, price, role)` for `good`."""
        stock_to_plan = max(1, expected_stock)
        extra_buffer = max(2, optimistic_stock - stock_to_plan + 6)
        num_goods = max(1, len(self._goods))
        inventory_cover = curr_inv / max(0.5, gp.ema_sales)

        endgame_discount = 1.0
        crowded_eg_window = 3 if turn > 40 else 6
        if is_crowded and remaining_turns <= crowded_eg_window and curr_inv >= 1:
            if remaining_turns == 6 and curr_inv >= 3:
                endgame_discount = 0.92
            elif remaining_turns == 5 and curr_inv >= 2:
                endgame_discount = 0.88
            elif remaining_turns == 4:
                endgame_discount = 0.84 if curr_inv >= 2 else 0.90
            elif remaining_turns == 3:
                endgame_discount = 0.78 if curr_inv >= 2 else 0.85
            elif remaining_turns == 2:
                endgame_discount = 0.70 if stock_to_plan > 1 else 0.80
            elif remaining_turns == 1:
                endgame_discount = 0.58 if stock_to_plan > 1 else 0.68
        elif remaining_turns <= (2 if turn > 40 else 3):
            if remaining_turns == 3 and (stock_to_plan > ceil(gp.ema_sales * 1.5) or curr_inv >= 2):
                endgame_discount = 0.88
            elif remaining_turns == 2:
                endgame_discount = 0.80 if stock_to_plan > 1 else 0.88
            elif remaining_turns == 1:
                endgame_discount = 0.68 if stock_to_plan > 1 else 0.76
        elif is_crowded and (crowding >= 0.40 or (comp_sellers >= 4 and inventory_cover >= 1.35)) and curr_inv >= 2:
            # Objective 3: Proactive clearance discount on crowded goods with shelf backlog.
            endgame_discount = max(0.80, 0.93 - 0.03 * min(4, curr_inv))
        elif inventory_cover > 2.6 and curr_inv >= 3:
            endgame_discount = max(0.82, 1.0 - 0.04 * (inventory_cover - 2.2))

        anchor = max(cost_floor, base_anchor * endgame_discount)
        if remaining_turns == 1 and gp.ema_sold_price > 0 and num_goods > 1:
            cap_mult = 0.70 if is_crowded else 0.78
            anchor = max(cost_floor, min(anchor, gp.ema_sold_price * cap_mult))

        ceiling = gp.price_ceiling if (gp.price_ceiling > cost_floor and remaining_turns > 1) else 0.0
        if ceiling > 0 and anchor > ceiling * 0.97:
            anchor = max(cost_floor, ceiling * 0.96)

        # Case 1: Only 1 expected unit available in Phase 3.
        if stock_to_plan == 1:
            if gp.zero_sale_streak >= 1 or (is_crowded and crowding >= 0.40 and curr_inv == 1):
                first_price = max(cost_floor, anchor * 0.92)
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
        # Dynamic Softmax Shelf-Space Scaling:
        # In crowded tournament rooms (10-44 teams) and long high-volume markets (240 turns),
        # splitting real inventory into 1-unit price-discriminating steps maximizes legitimate
        # softmax shelf share against multi-listing rivals while staying well under 20 ms.
        active_qty = stock_to_plan
        if is_expansion:
            max_steps = min(active_qty, 50)
        elif is_crowded:
            max_steps = min(active_qty, 28)
        else:
            max_steps = 12 if num_goods == 1 else 8
        chunk = max(1, ceil(active_qty / max(1, max_steps)))
        tiers: list[tuple[int, float, str]] = []

        crowded_clearance = (
            is_crowded
            and (crowding >= 0.40 or (comp_sellers >= 4 and inventory_cover >= 1.35))
            and curr_inv >= 2
        )
        wide_multi_segment = (
            crowded_clearance
            or (num_goods == 1 and gp.init_endowment >= 10 and active_qty >= 4)
            or (active_qty >= 5 and comp_stock > 0)
            or (curr_inv >= 3 and inventory_cover > 2.0 and comp_stock > 0)
            or (is_crowded and crowding >= 0.35 and curr_inv >= 1)
        )

        if ceiling > 0:
            if crowded_clearance:
                step_mults = [0.78, 0.84, 0.89, 0.94, 0.98, 1.00, 1.02]
            elif wide_multi_segment or curr_inv >= 3:
                step_mults = [0.80, 0.86, 0.91, 0.95, 0.98, 1.00, 1.02]
            else:
                step_mults = [0.94, 0.97, 0.99, 1.00, 1.015, 1.03]
        elif not gp.has_ever_sold and turn == 1:
            if num_goods == 1 and gp.init_endowment == 0:
                step_mults = [0.95, 0.97, 0.99, 1.01, 1.03, 1.06]
            elif is_crowded:
                step_mults = [0.86, 0.92, 0.97, 1.02, 1.08, 1.15]
            elif wide_multi_segment:
                step_mults = [0.78, 0.84, 0.90, 0.96, 1.01, 1.06, 1.12]
            else:
                step_mults = [0.92, 0.97, 1.02, 1.07, 1.13, 1.20]
        elif crowded_clearance:
            step_mults = [0.78, 0.84, 0.90, 0.95, 1.00, 1.05]
        elif gp.zero_sale_streak >= 1 or (curr_inv >= 3 and inventory_cover > 2.2):
            step_mults = [0.80, 0.86, 0.91, 0.96, 1.00, 1.04]
        elif is_crowded and comp_sellers <= 1 and is_blitz:
            # Blitz monopoly staircase: extract full top-segment WTP.
            step_mults = [0.98, 1.02, 1.06, 1.11, 1.16, 1.22]
        elif wide_multi_segment:
            step_mults = [0.82, 0.88, 0.94, 0.99, 1.03, 1.08, 1.14]
        else:
            step_mults = [0.95, 0.99, 1.02, 1.05, 1.09, 1.14]

        if demand_heat > 1.25 and gp.zero_sale_streak == 0 and inventory_cover < 1.8:
            step_mults = [m * 1.025 for m in step_mults]

        allocated = 0
        step_idx = 0
        total_steps = max(1, ceil(active_qty / chunk))
        while allocated < active_qty:
            qty = min(chunk, active_qty - allocated)
            frac = step_idx / max(1, total_steps - 1)
            pos = frac * (len(step_mults) - 1)
            lo_idx = int(pos)
            hi_idx = min(lo_idx + 1, len(step_mults) - 1)
            w = pos - lo_idx
            mult = step_mults[lo_idx] * (1.0 - w) + step_mults[hi_idx] * w

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
        if ceiling > 0:
            catchall_price = max(cost_floor, min(top_active_price * 1.01, ceiling * 0.99))
        else:
            catchall_price = max(top_active_price * 1.02, anchor * 1.05)
        tiers.append((extra_buffer, catchall_price, "probe"))
        return tiers

    # ------------------------------------------------------------------
    # Production & cost helpers (integrated with MarketContext v2/v3)
    # ------------------------------------------------------------------

    def _estimate_current_production(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        """Simulate `ctx.planned_assembly` under 0%, expected, and 100% bid fill rates."""
        lot_caps = {lot.good: max(0, int(lot.quantity)) for lot in state.auction_lots}
        lot_reserves = {lot.good: float(lot.reserve_price) for lot in state.auction_lots}
        bid_map: dict[str, int] = {}
        for bid in ctx.planned_bids:
            if bid.quantity > 0 and bid.price_per_unit >= lot_reserves.get(bid.good, 0.0) - 1e-9:
                cap = lot_caps.get(bid.good, bid.quantity)
                bid_map[bid.good] = min(cap, bid_map.get(bid.good, 0) + int(bid.quantity))

        fill_map = getattr(ctx, "auction_fill_rate", {})
        inv_min = {raw: max(0, int(state.inventory.get(raw, 0))) for raw in goods_config.raw_materials}
        inv_exp = {
            raw: inv_min[raw] + int(round(fill_map.get(raw, 0.70) * bid_map.get(raw, 0)))
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
        rnames = self._good_to_recipes.get(good, [])
        if not rnames:
            return gp.init_cost

        lot_prices = {lot.good: lot.reserve_price for lot in state.auction_lots}
        best_cost = float("inf")
        for rname in rnames:
            if rname not in goods_config.recipes:
                continue
            ctx_cost = ctx.estimate_recipe_cost(rname, goods_config)
            recipe = goods_config.recipes[rname]
            reserve_cost = sum(
                lot_prices.get(mat, ctx.avg_material_cost.get(mat, 0.0)) * qty
                for mat, qty in recipe.inputs.items()
            )
            best_cost = min(best_cost, max(ctx_cost, reserve_cost))

        if not isfinite(best_cost):
            best_cost = gp.init_cost
        return max(0.01, best_cost, gp.init_cost * 0.75)

    @staticmethod
    def _compute_cost_floor(
        unit_cost: float,
        gp: _GoodPricingState,
        curr_inv: int,
        remaining_turns: int,
        is_crowded: bool = False,
    ) -> float:
        if remaining_turns <= 1 and curr_inv > 0:
            return max(0.02, unit_cost * (0.45 if is_crowded else 0.60))
        if remaining_turns == 2 and curr_inv >= 1:
            return max(0.02, unit_cost * (0.70 if is_crowded else 0.82))
        if is_crowded and remaining_turns <= 4 and curr_inv >= 2:
            return max(0.02, unit_cost * 0.88)
        return max(0.02, unit_cost * 1.12)

    @staticmethod
    def _competitor_finished_stock(
        state: Observation, ctx: MarketContext, good: str
    ) -> int:
        comp_inv_map = getattr(ctx, "competitor_finished_inventory", None)
        if comp_inv_map and good in comp_inv_map:
            return max(0, int(comp_inv_map[good]))
        total = 0
        for tid, inv in state.public_inventories.items():
            if ctx.team_id is not None and tid == ctx.team_id:
                continue
            if state.public_capital.get(tid, 0.0) < 0.0:
                continue
            total += max(0, int(inv.get(good, 0)))
        if ctx.team_id is None:
            total = max(0, total - max(0, int(state.inventory.get(good, 0))))
        return total
