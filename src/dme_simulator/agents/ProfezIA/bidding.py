"""Phase 1: Raw-material auction bidding strategy (Owner: Persona 1).

Keep all bidding logic and any bidding-specific internal state inside this
module (e.g. attributes on `self`) so changes never conflict with other files.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

from dme_simulator.agents import Bid, Observation
from dme_simulator.config import GoodsConfig

from .state import MarketContext


@dataclass
class _MaterialBidState:
    """Internal per-material tracking for pay-your-bid auction optimization."""

    init_reserve: float = 10.0
    last_bid_qty: int = 0
    last_bid_price: float = 0.0
    last_regime: int = 1
    prev_raw_inv: int = 0
    ema_consumed: float = 1.0
    ema_fill_rate: float = 1.0
    demand_scale: float = 1.0
    mid_mult: float = 1.303
    high_mult: float = 1.453


@dataclass
class _RecipeTrackState:
    """Internal per-recipe tracking to coordinate with Phase 2 production capacity."""

    prev_finished_inv: int = 0
    ema_produced: float = 1.0
    ema_sold: float = 1.0
    inferred_prod_cap: int = 4
    is_hard_capped: bool = False


class BiddingStrategy:
    """Adaptive sealed-bid strategy with regime-based clearing estimation and bottleneck balancing."""

    def __init__(self) -> None:
        self._initialized: bool = False
        self._starting_capital: float = 10000.0
        self._est_total_turns: int = 40
        self._materials: dict[str, _MaterialBidState] = {}
        self._recipes: dict[str, _RecipeTrackState] = {}
        self._init_team_capital: dict[str, float] = {}
        self._init_team_inv: dict[str, dict[str, int]] = {}
        self._active_opponents: set[str] = set()

    def decide_bids(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> list[Bid]:
        """Return sealed bids for this turn's `state.auction_lots`."""
        if not self._initialized or state.turn == 1:
            self._init_turn_one(state, goods_config, ctx)
        else:
            self._update_from_turn_results(state, goods_config, ctx)

        if state.turn >= self._est_total_turns:
            self._est_total_turns = state.turn + 2

        remaining_turns = max(1, self._est_total_turns - state.turn + 1)
        if state.capital <= 1.0 or not state.auction_lots:
            self._record_pre_turn_state(state, goods_config, {})
            return []

        # 1. Rank recipes by expected net margin and determine target batch quantities.
        recipe_targets, max_material_wtp = self._plan_recipe_material_targets(
            state, goods_config, ctx, remaining_turns
        )

        # 2. Convert recipe targets into net raw-material requirements (subtracting current inventory).
        raw_needed: dict[str, int] = {raw: 0 for raw in goods_config.raw_materials}
        material_priority: dict[str, float] = {raw: 0.0 for raw in goods_config.raw_materials}

        for rname, (target_qty, unit_margin) in recipe_targets.items():
            if target_qty <= 0 or rname not in goods_config.recipes:
                continue
            recipe = goods_config.recipes[rname]
            for mat, amt in recipe.inputs.items():
                if amt <= 0:
                    continue
                raw_needed[mat] = raw_needed.get(mat, 0) + int(target_qty) * int(amt)
                material_priority[mat] = max(
                    material_priority.get(mat, 0.0),
                    unit_margin / max(1, int(amt)),
                )

        # Incorporate shadow value signals from `ctx` if available.
        ctx_shadows = getattr(ctx, "material_shadow_value", {})
        for mat, sval in ctx_shadows.items():
            if sval > 0:
                material_priority[mat] = max(material_priority.get(mat, 0.0), float(sval))

        # 3. Price each material lot using the 3-regime descending clearing model.
        planned_bids: dict[str, tuple[int, float, int, float]] = {}
        for lot in state.auction_lots:
            mat = lot.good
            if lot.quantity <= 0 or mat not in self._materials:
                continue

            curr_raw = max(0, int(state.inventory.get(mat, 0)))
            gross_target = raw_needed.get(mat, 0)
            net_qty = max(0, gross_target - curr_raw)

            # Cap inventory accumulation so we never hoard unused raw materials.
            ms = self._materials[mat]
            max_raw_stock = max(
                gross_target,
                int(ceil(max(1.0, ms.ema_consumed) * min(float(remaining_turns), 2.2))),
            )
            if curr_raw >= max_raw_stock:
                net_qty = 0
            else:
                net_qty = min(net_qty, max_raw_stock - curr_raw, int(lot.quantity))

            if net_qty <= 0:
                continue

            bid_price, regime, adj_qty = self._compute_optimal_bid(
                lot=lot,
                net_qty=net_qty,
                max_wtp=max_material_wtp.get(mat, lot.reserve_price * 2.5),
                state=state,
                goods_config=goods_config,
                ctx=ctx,
                remaining_turns=remaining_turns,
            )
            if adj_qty <= 0 or bid_price < lot.reserve_price or not isfinite(bid_price):
                continue

            planned_bids[mat] = (
                adj_qty,
                bid_price,
                regime,
                material_priority.get(mat, 1.0),
            )

        # 4. Enforce strict budget cap so random lot clearing order never starves key materials.
        final_bids = self._enforce_budget_cap(
            planned_bids, state, goods_config, remaining_turns
        )

        self._record_pre_turn_state(state, goods_config, final_bids)
        return [
            Bid(mat, int(qty), round(float(price), 4))
            for mat, (qty, price, _regime) in final_bids.items()
            if qty > 0 and isfinite(price) and price >= 0.0
        ]

    # ------------------------------------------------------------------
    # Turn 1 setup & Turn t>=2 learning
    # ------------------------------------------------------------------

    def _init_turn_one(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> None:
        self._initialized = True
        self._starting_capital = max(1.0, float(state.capital))

        total_initial_lots = sum(max(0, int(lot.quantity)) for lot in state.auction_lots)
        num_finished = max(1, len(goods_config.finished_goods))
        if num_finished <= 2 and self._starting_capital <= 8000.0:
            self._est_total_turns = 20
        elif self._starting_capital >= 30000.0 or total_initial_lots >= 500:
            self._est_total_turns = 30
        else:
            self._est_total_turns = 40

        self._materials.clear()
        lot_map = {lot.good: lot for lot in state.auction_lots}
        for raw in goods_config.raw_materials:
            lot = lot_map.get(raw)
            rprice = max(0.01, float(lot.reserve_price)) if lot else 10.0
            base_need = sum(
                int(r.inputs.get(raw, 0)) for r in goods_config.recipes.values()
            )
            self._materials[raw] = _MaterialBidState(
                init_reserve=rprice,
                prev_raw_inv=max(0, int(state.inventory.get(raw, 0))),
                ema_consumed=float(max(1, base_need * 2)),
            )

        avg_inputs = (
            sum(sum(r.inputs.values()) for r in goods_config.recipes.values())
            / max(1, len(goods_config.recipes))
        )
        team_count = max(2, len(state.public_capital))
        est_market_good_units = total_initial_lots / max(
            1.0, avg_inputs * team_count * num_finished
        )
        init_cap = max(2, min(10, int(ceil(est_market_good_units * 1.5))))
        if num_finished == 1:
            init_cap = max(init_cap, 10)

        self._recipes.clear()
        for rname, recipe in goods_config.recipes.items():
            g = recipe.output_good
            self._recipes[rname] = _RecipeTrackState(
                prev_finished_inv=max(0, int(state.inventory.get(g, 0))),
                ema_produced=float(init_cap),
                ema_sold=float(init_cap),
                inferred_prod_cap=init_cap,
                is_hard_capped=False,
            )

        self._init_team_capital = {
            tid: float(cap) for tid, cap in state.public_capital.items()
        }
        self._init_team_inv = {
            tid: dict(inv) for tid, inv in state.public_inventories.items()
        }
        self._active_opponents = {
            tid
            for tid in state.public_capital
            if ctx.team_id is None or tid != ctx.team_id
        }

    def _update_from_turn_results(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> None:
        # 1. Identify active opponent teams (exclude passive agents like PatientAgent).
        active: set[str] = set()
        for tid, cap in state.public_capital.items():
            if ctx.team_id is not None and tid == ctx.team_id:
                continue
            init_cap = self._init_team_capital.get(tid, cap)
            init_inv = self._init_team_inv.get(tid, {})
            curr_inv = state.public_inventories.get(tid, {})
            if abs(cap - init_cap) > 1e-3 or curr_inv != init_inv:
                active.add(tid)
        if state.turn >= 2:
            self._active_opponents = active

        # 2. Tally our auction wins from turn t-1.
        won_by_mat: dict[str, int] = {raw: 0 for raw in goods_config.raw_materials}
        for tx in ctx.new_auction_txs:
            if tx.quantity <= 0:
                continue
            if ctx.team_id is None or tx.buyer == ctx.team_id:
                won_by_mat[tx.good] = won_by_mat.get(tx.good, 0) + int(tx.quantity)

        # 3. Update per-material consumption and auction fill-rate adaptation.
        for raw, ms in self._materials.items():
            curr_raw = max(0, int(state.inventory.get(raw, 0)))
            won_qty = won_by_mat.get(raw, 0)
            actual_consumed = max(0, ms.prev_raw_inv + won_qty - curr_raw)
            ms.ema_consumed = 0.45 * actual_consumed + 0.55 * ms.ema_consumed

            if ms.last_bid_qty > 0:
                fill_rate = min(1.0, won_qty / float(ms.last_bid_qty))
                ms.ema_fill_rate = 0.45 * fill_rate + 0.55 * ms.ema_fill_rate

                r_hist = ctx.reserve_price_history.get(raw, [])
                reserve_fell = (
                    len(r_hist) >= 2 and r_hist[-1] < r_hist[-2] * 0.995
                )

                if fill_rate < 0.85:
                    ms.demand_scale = min(1.65, ms.demand_scale + 0.22)
                    if ms.last_regime == 2:
                        ms.mid_mult = min(1.38, ms.mid_mult + 0.015)
                    elif ms.last_regime == 3:
                        ms.high_mult = min(1.65, ms.high_mult + 0.035)
                elif fill_rate >= 0.99 and reserve_fell:
                    ms.demand_scale = max(0.70, ms.demand_scale * 0.92)
                    ms.high_mult = max(1.453, ms.high_mult * 0.99)

        # 4. Update per-recipe production & sales tracking.
        sold_by_good: dict[str, int] = {g: 0 for g in goods_config.finished_goods}
        for stx in ctx.new_sales_txs:
            if stx.quantity <= 0:
                continue
            if ctx.team_id is None or stx.seller == ctx.team_id:
                sold_by_good[stx.good] = sold_by_good.get(stx.good, 0) + int(stx.quantity)

        for rname, recipe in goods_config.recipes.items():
            rt = self._recipes.get(rname)
            if rt is None:
                continue
            g = recipe.output_good
            curr_fin = max(0, int(state.inventory.get(g, 0)))
            sold_qty = sold_by_good.get(g, 0)

            actual_produced = max(0, curr_fin + sold_qty - rt.prev_finished_inv)
            rt.ema_produced = 0.45 * actual_produced + 0.55 * rt.ema_produced
            rt.ema_sold = 0.45 * sold_qty + 0.55 * rt.ema_sold

            leftover_sets = min(
                (
                    max(0, int(state.inventory.get(mat, 0))) // max(1, int(amt))
                    for mat, amt in recipe.inputs.items()
                ),
                default=0,
            )
            if leftover_sets >= 1 and actual_produced <= 1 and curr_fin == 0:
                # Only if shelf was empty and production still stopped at 1 is it a static 1-unit stub.
                rt.inferred_prod_cap = max(1, actual_produced)
                rt.is_hard_capped = True
            elif actual_produced > 1:
                rt.is_hard_capped = False
                rt.inferred_prod_cap = max(rt.inferred_prod_cap, actual_produced + 2)

    # ------------------------------------------------------------------
    # Recipe profitability & target planning
    # ------------------------------------------------------------------

    def _plan_recipe_material_targets(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
        remaining_turns: int,
    ) -> tuple[dict[str, tuple[int, float]], dict[str, float]]:
        """Compute `(target_units, unit_margin)` per recipe and `max_wtp` per material."""
        lot_reserve = {lot.good: float(lot.reserve_price) for lot in state.auction_lots}
        lot_qty_map = {lot.good: int(lot.quantity) for lot in state.auction_lots}
        est_prices = self._estimate_all_good_prices(state, goods_config, ctx)

        recipe_targets: dict[str, tuple[int, float]] = {}
        max_material_wtp: dict[str, float] = {
            raw: lot_reserve.get(raw, 10.0) * 1.48 for raw in goods_config.raw_materials
        }

        ranked_recipes: list[tuple[float, str]] = []
        for rname, recipe in goods_config.recipes.items():
            g = recipe.output_good
            sale_p = est_prices.get(g, 150.0)
            est_cost = sum(
                max(
                    lot_reserve.get(mat, 1.0),
                    ctx.avg_material_cost.get(mat, lot_reserve.get(mat, 1.0)),
                )
                * int(amt)
                for mat, amt in recipe.inputs.items()
            )
            margin = sale_p - est_cost
            ranked_recipes.append((margin, rname))

            for focal_mat, focal_amt in recipe.inputs.items():
                if focal_amt <= 0:
                    continue
                other_cost = sum(
                    max(
                        lot_reserve.get(m, 1.0),
                        ctx.avg_material_cost.get(m, lot_reserve.get(m, 1.0)),
                    )
                    * int(a)
                    for m, a in recipe.inputs.items()
                    if m != focal_mat
                )
                max_unit_wtp = (sale_p * 0.85 - other_cost) / float(focal_amt)
                if max_unit_wtp > max_material_wtp.get(focal_mat, 0.0):
                    max_material_wtp[focal_mat] = max_unit_wtp

        ranked_recipes.sort(reverse=True)
        num_finished = max(1, len(goods_config.finished_goods))

        own_sold_ema_map = getattr(ctx, "own_sold_ema", {})
        own_sold_last_map = getattr(ctx, "own_sold_last_turn", {})
        stockout_map = getattr(ctx, "own_stockout_streak", {})
        unsold_map = getattr(ctx, "own_unsold_streak", {})
        sales_count_map = getattr(ctx, "sales_count_by_good", {})

        for margin, rname in ranked_recipes:
            recipe = goods_config.recipes[rname]
            rt = self._recipes.get(rname)
            g = recipe.output_good
            curr_fin = max(0, int(state.inventory.get(g, 0)))

            if margin <= 0.0 or rt is None:
                recipe_targets[rname] = (0, 0.0)
                continue

            if rt.is_hard_capped:
                target_u = max(0, rt.inferred_prod_cap - curr_fin)
                recipe_targets[rname] = (target_u, margin)
                continue

            vel = float(own_sold_ema_map.get(g, rt.ema_sold))
            last_sold = int(own_sold_last_map.get(g, 0))
            stockouts = int(stockout_map.get(g, 0))
            unsold = int(unsold_map.get(g, 0))
            total_sold = int(sales_count_map.get(g, 0))

            if num_finished == 1:
                # Single-good market: check how many units the lot can supply after opponents.
                if state.turn == 1 and curr_fin >= 10:
                    # Large initial finished inventory (e.g. tutorial_pricing): buy lightly.
                    desired_total = curr_fin + 2
                else:
                    # Estimate uncontested + affordable units in lot (e.g. tutorial_auction).
                    max_from_lot = min(
                        lot_qty_map.get(m, 0) // max(1, int(a))
                        for m, a in recipe.inputs.items()
                    )
                    if total_sold == 0:
                        desired_total = min(10, max(4, max_from_lot))
                    elif stockouts >= 1:
                        desired_total = max(
                            int(ceil(vel * 1.6 + 2.0)),
                            int(ceil(last_sold * 1.4 + 2.0)),
                        )
                    else:
                        desired_total = max(2, int(ceil(vel * 1.3 + 1.0)))
            else:
                # Multi-good market: align with `production.py` target shelf stocks & velocity.
                if total_sold == 0:
                    if unsold >= 2 or curr_fin >= 3:
                        desired_total = curr_fin
                    else:
                        desired_total = max(2, min(4, rt.inferred_prod_cap))
                elif unsold >= 2:
                    desired_total = max(0, int(vel * 0.6))
                elif unsold == 1:
                    desired_total = max(1, int(ceil(vel * 0.95)))
                elif stockouts >= 1:
                    desired_total = max(
                        int(ceil(vel * 1.65 + 1.5)),
                        int(ceil(last_sold * 1.4 + 1.0)),
                    )
                else:
                    desired_total = max(1, int(ceil(vel * 1.3 + 0.5)))

            # Endgame taper: never buy materials for more units than remaining turns can absorb.
            if remaining_turns <= 4:
                eff_vel = max(0.8, vel) if total_sold > 0 else 0.5
                if remaining_turns == 1:
                    endgame_max = int(eff_vel * 0.85) if curr_fin == 0 else 0
                elif remaining_turns == 2:
                    endgame_max = int(ceil(eff_vel * 1.55))
                else:
                    endgame_max = int(ceil(eff_vel * (remaining_turns - 0.5)))
                desired_total = min(desired_total, endgame_max)

            net_u = max(0, desired_total - curr_fin)
            recipe_targets[rname] = (net_u, margin)

        return recipe_targets, max_material_wtp

    def _estimate_all_good_prices(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> dict[str, float]:
        """Return estimated sale price for every finished good."""
        init_costs: dict[str, float] = {}
        input_counts: dict[str, int] = {}
        for rname, recipe in goods_config.recipes.items():
            g = recipe.output_good
            c = sum(
                self._materials[mat].init_reserve * int(amt)
                for mat, amt in recipe.inputs.items()
                if mat in self._materials
            )
            u = sum(int(amt) for amt in recipe.inputs.values())
            init_costs[g] = max(0.5, c)
            input_counts[g] = max(1, u)

        c_min = min(init_costs.values()) if init_costs else 1.0
        c_max = max(init_costs.values()) if init_costs else 1.0
        u_min = min(input_counts.values()) if input_counts else 1
        u_max = max(input_counts.values()) if input_counts else 1
        num_finished = len(goods_config.finished_goods)

        est: dict[str, float] = {}
        for g in goods_config.finished_goods:
            icost = init_costs.get(g, 25.0)
            iunits = input_counts.get(g, 2)
            if num_finished > 1 and (c_max - c_min > 1e-6 or u_max > u_min):
                r_cost = (icost - c_min) / max(1e-6, c_max - c_min)
                r_units = (iunits - u_min) / max(1e-6, float(u_max - u_min))
                r_tier = 0.72 * r_cost + 0.28 * r_units
                prior = icost * (5.0 + 10.0 * (r_tier ** 1.2))
            else:
                if icost < self._starting_capital * 0.002:
                    prior = max(icost * 5.0, self._starting_capital * 0.025)
                else:
                    prior = icost * 5.0

            observed = ctx.est_sale_price.get(g, 0.0)
            est[g] = observed if observed > icost * 1.2 else prior
        return est

    # ------------------------------------------------------------------
    # 3-Regime Pay-Your-Bid Clearing Price Model
    # ------------------------------------------------------------------

    def _compute_optimal_bid(
        self,
        lot,
        net_qty: int,
        max_wtp: float,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
        remaining_turns: int,
    ) -> tuple[float, int, int]:
        """Select optimal `(bid_price, regime, adj_qty)` for `lot`."""
        mat = lot.good
        rprice = float(lot.reserve_price)
        lot_qty = int(lot.quantity)
        ms = self._materials[mat]

        d_high, d_mid = self._estimate_opponent_demand(
            mat, rprice, state, goods_config, ctx
        )
        d_high_scaled = int(ceil(d_high * ms.demand_scale))
        d_mid_scaled = int(ceil(d_mid * ms.demand_scale))

        free_after_all = max(0, lot_qty - d_high_scaled - d_mid_scaled)
        free_after_high = max(0, lot_qty - d_high_scaled)

        # If the uncontested surplus in the lot already satisfies most of our need
        # (or in a single-good auction where uncontested units are abundant),
        # cap quantity to `free_after_all` and win 100% at the reserve price!
        if free_after_all >= net_qty:
            regime = 1
            raw_price = rprice * 1.0012
            adj_qty = net_qty
        elif (
            len(goods_config.finished_goods) == 1
            and free_after_all >= max(2, int(net_qty * 0.65))
        ):
            regime = 1
            raw_price = rprice * 1.0012
            adj_qty = free_after_all
        elif free_after_high >= net_qty:
            regime = 2
            raw_price = rprice * ms.mid_mult
            adj_qty = net_qty
        else:
            regime = 3
            raw_price = rprice * ms.high_mult
            adj_qty = net_qty

        # On the final turn, avoid paying high-contest premiums.
        if remaining_turns == 1 and regime == 3 and len(goods_config.finished_goods) > 1:
            if free_after_high > 0:
                regime = 2
                raw_price = rprice * ms.mid_mult
                adj_qty = min(net_qty, free_after_high)
            else:
                return -1.0, regime, 0

        capped_price = min(raw_price, max_wtp)
        if capped_price < rprice:
            return -1.0, regime, 0
        return max(rprice * 1.0005, capped_price), regime, adj_qty

    def _estimate_opponent_demand(
        self,
        mat: str,
        rprice: float,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> tuple[int, int]:
        """Estimate `(d_high, d_mid)` opponent demand for `mat` from public state."""
        one_batch_need = sum(
            int(recipe.inputs.get(mat, 0))
            for recipe in goods_config.recipes.values()
        )
        if one_batch_need <= 0:
            return 0, 0

        opp_ids = [
            tid
            for tid in state.public_inventories
            if (ctx.team_id is None or tid != ctx.team_id)
            and (state.turn == 1 or tid in self._active_opponents)
            and state.public_capital.get(tid, 0.0) > rprice
        ]
        if not opp_ids:
            return 0, 0

        lot_reserves = {lot.good: float(lot.reserve_price) for lot in state.auction_lots}
        one_batch_total_cost = sum(
            lot_reserves.get(m, 10.0)
            * sum(int(r.inputs.get(m, 0)) for r in goods_config.recipes.values())
            for m in goods_config.raw_materials
        )

        demands: list[tuple[int, int]] = []
        for tid in opp_ids:
            inv_m = max(0, int(state.public_inventories.get(tid, {}).get(mat, 0)))
            cap = max(0.0, float(state.public_capital.get(tid, 0.0)))

            mid_need = max(0, one_batch_need - inv_m)
            raw_high_need = max(0, 4 * one_batch_need - inv_m)
            if one_batch_total_cost > 0:
                planned_4x_cost = 4.0 * one_batch_total_cost * 1.45
                budget = 0.60 * cap
                scale = min(1.0, budget / planned_4x_cost) if planned_4x_cost > budget else 1.0
                high_need = int(raw_high_need * scale)
            else:
                high_need = raw_high_need
            demands.append((high_need, mid_need))

        if len(demands) == 1:
            return demands[0][0], 0

        demands.sort(key=lambda pair: pair[0], reverse=True)
        d_high = demands[0][0]
        d_mid = sum(pair[1] for pair in demands[1:])
        return d_high, d_mid

    # ------------------------------------------------------------------
    # Budget Cap Enforcement
    # ------------------------------------------------------------------

    def _enforce_budget_cap(
        self,
        planned_bids: dict[str, tuple[int, float, int, float]],
        state: Observation,
        goods_config: GoodsConfig,
        remaining_turns: int,
    ) -> dict[str, tuple[int, float, int]]:
        """Ensure total auction commitment stays safely within `state.capital`."""
        if not planned_bids:
            return {}

        budget_share = 0.78 if remaining_turns > 2 else (0.50 if remaining_turns == 2 else 0.30)
        max_budget = max(0.0, float(state.capital) * budget_share)

        total_cost = sum(qty * price for qty, price, _, _ in planned_bids.values())
        if total_cost <= max_budget + 1e-6:
            return {
                mat: (qty, price, regime)
                for mat, (qty, price, regime, _prio) in planned_bids.items()
            }

        scale = max_budget / max(1e-6, total_cost)
        scaled: dict[str, tuple[int, float, int]] = {}
        running_spend = 0.0

        ordered = sorted(
            planned_bids.items(),
            key=lambda item: item[1][3],
            reverse=True,
        )
        for mat, (qty, price, regime, _prio) in ordered:
            if price <= 0:
                continue
            rem_budget = max_budget - running_spend
            max_affordable = int(rem_budget // price)
            if max_affordable <= 0:
                continue
            target_q = max(1, int(qty * scale))
            final_q = min(qty, target_q, max_affordable)
            if final_q > 0:
                scaled[mat] = (final_q, price, regime)
                running_spend += final_q * price

        return scaled

    def _record_pre_turn_state(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        final_bids: dict[str, tuple[int, float, int]],
    ) -> None:
        for raw, ms in self._materials.items():
            ms.prev_raw_inv = max(0, int(state.inventory.get(raw, 0)))
            if raw in final_bids:
                qty, price, regime = final_bids[raw]
                ms.last_bid_qty = int(qty)
                ms.last_bid_price = float(price)
                ms.last_regime = int(regime)
            else:
                ms.last_bid_qty = 0
                ms.last_bid_price = 0.0
                ms.last_regime = 0

        for rname, recipe in goods_config.recipes.items():
            rt = self._recipes.get(rname)
            if rt is not None:
                rt.prev_finished_inv = max(
                    0, int(state.inventory.get(recipe.output_good, 0))
                )
