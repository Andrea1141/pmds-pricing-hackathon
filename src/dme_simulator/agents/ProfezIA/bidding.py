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


_HORIZON_CHECKPOINTS = (20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 240)


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
    zero_bid_price: float = 0.65


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
        self._total_init_lots: int = 0
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

        if state.turn > self._est_total_turns:
            next_cp = next((cp for cp in _HORIZON_CHECKPOINTS if cp >= state.turn), None)
            self._est_total_turns = next_cp if next_cp is not None else state.turn + 40

        remaining_turns = max(1, self._est_total_turns - state.turn + 1)
        if state.capital <= 0.01 or not state.auction_lots:
            self._record_pre_turn_state(state, goods_config, {})
            return []

        # 1. Rank recipes by expected net margin and determine target batch quantities.
        recipe_targets, max_material_wtp, est_prices = self._plan_recipe_material_targets(
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

        # Precompute opponent-demand total batch cost once per turn.
        lot_reserves = {lot.good: float(lot.reserve_price) for lot in state.auction_lots}
        one_batch_need_by_mat = self._compute_one_batch_need(goods_config)
        one_batch_total_cost = sum(
            lot_reserves.get(m, 10.0) * one_batch_need_by_mat.get(m, 0)
            for m in goods_config.raw_materials
        )

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
                one_batch_need=one_batch_need_by_mat.get(mat, 0),
                one_batch_total_cost=one_batch_total_cost,
                lot_reserves=lot_reserves,
            )
            if adj_qty <= 0 or bid_price < lot.reserve_price or not isfinite(bid_price):
                continue

            planned_bids[mat] = (
                adj_qty,
                bid_price,
                regime,
                material_priority.get(mat, 1.0),
            )

        # 3b. Enforce joint recipe-bundle ROI discipline so high multipliers on multiple
        # inputs never combine to exceed expected finished-good selling prices.
        planned_bids = self._enforce_bundle_roi_caps(
            planned_bids=planned_bids,
            recipe_targets=recipe_targets,
            est_prices=est_prices,
            lot_reserves=lot_reserves,
            state=state,
            goods_config=goods_config,
            ctx=ctx,
        )

        # 4. Enforce strict budget cap so random lot clearing order never starves key materials.
        final_bids = self._enforce_budget_cap(
            planned_bids, state, goods_config, ctx, remaining_turns
        )

        self._record_pre_turn_state(state, goods_config, final_bids)
        out_bids: list[Bid] = []
        for mat, (qty, price, _regime) in final_bids.items():
            if qty <= 0 or not isfinite(price) or price < 0.0:
                continue
            rprice = lot_reserves.get(mat, 0.0)
            rounded = round(float(price), 6)
            if rounded < rprice:
                rounded = rprice
            out_bids.append(Bid(mat, int(qty), rounded))
        return out_bids

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
        self._total_init_lots = total_initial_lots
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
            rprice = max(0.0, float(lot.reserve_price)) if lot else 10.0
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
        is_own_auc = getattr(ctx, "is_own_auction_tx", None)
        for tx in ctx.new_auction_txs:
            if tx.quantity <= 0:
                continue
            if is_own_auc(tx) if is_own_auc is not None else (ctx.team_id is not None and tx.buyer == ctx.team_id):
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
                prev_reserve = r_hist[-2] if len(r_hist) >= 2 else (r_hist[-1] if r_hist else 1.0)
                if prev_reserve <= 1e-6:
                    if fill_rate < 0.85 and ms.last_regime >= 2:
                        step = 1.45 if fill_rate < 0.50 else 1.25
                        ms.zero_bid_price = max(ms.zero_bid_price, ms.last_bid_price) * step
                    elif fill_rate >= 0.95 and ms.last_bid_price > 0.0:
                        ms.zero_bid_price = max(0.10, ms.last_bid_price * 0.94)

                if fill_rate < 0.85:
                    severe = fill_rate < 0.50
                    ms.demand_scale = min(
                        2.40, ms.demand_scale + (0.35 if severe else 0.22)
                    )
                    if ms.last_regime == 2:
                        ms.mid_mult = min(
                            2.10, ms.mid_mult + (0.05 if severe else 0.025)
                        )
                    elif ms.last_regime == 3:
                        ms.high_mult = min(
                            4.20, ms.high_mult + (0.12 if severe else 0.05)
                        )
                elif fill_rate >= 0.99 and reserve_fell:
                    ms.demand_scale = max(0.70, ms.demand_scale * 0.92)
                    ms.mid_mult = max(1.303, ms.mid_mult * 0.99)
                    ms.high_mult = max(1.453, ms.high_mult * 0.985)

        # 4. Update per-recipe production & sales tracking.
        sold_by_good: dict[str, int] = {g: 0 for g in goods_config.finished_goods}
        is_own_sale = getattr(ctx, "is_own_sales_tx", None)
        for stx in ctx.new_sales_txs:
            if stx.quantity <= 0:
                continue
            if is_own_sale(stx) if is_own_sale is not None else (ctx.team_id is not None and stx.seller == ctx.team_id):
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
            # Rely on `ctx.estimated_capacity_per_turn` instead of brittle per-recipe
            # hard-cap heuristics that can permanently lock a recipe to 1 unit/turn.
            rt.is_hard_capped = False
            if actual_produced > 1:
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
    ) -> tuple[dict[str, tuple[int, float]], dict[str, float], dict[str, float]]:
        """Compute `(target_units, unit_margin)` per recipe, `max_wtp` per material, and `est_prices`."""
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
                    lot_reserve.get(mat, 0.0),
                    ctx.avg_material_cost.get(mat, lot_reserve.get(mat, 0.0)),
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
                        lot_reserve.get(m, 0.0),
                        ctx.avg_material_cost.get(m, lot_reserve.get(m, 0.0)),
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
        crowding_map = getattr(ctx, "crowding_ratio_by_good", {})
        supply_per_team = float(getattr(ctx, "supply_per_team", 10.0))
        is_crowded = bool(getattr(ctx, "is_crowded_tournament", False))
        is_expansion = bool(getattr(ctx, "is_high_volume_expansion", False))
        is_blitz = bool(getattr(ctx, "is_blitz_window", False))
        est_cap = getattr(ctx, "estimated_capacity_per_turn", None)
        remaining_cap = int(est_cap) if est_cap is not None and est_cap > 0 else None
        covered_goods: set[str] = set()

        # Identify top-2 highest-margin unique goods for blitz focus.
        blitz_top_goods: set[str] = set()
        if is_blitz and num_finished > 1:
            for m_val, rn in ranked_recipes:
                if m_val <= 0.0:
                    continue
                g_out = goods_config.recipes[rn].output_good
                blitz_top_goods.add(g_out)
                if len(blitz_top_goods) >= 2:
                    break

        for margin, rname in ranked_recipes:
            recipe = goods_config.recipes[rname]
            rt = self._recipes.get(rname)
            g = recipe.output_good
            curr_fin = max(0, int(state.inventory.get(g, 0)))

            # On multi-recipe goods, only plan material purchases for the highest-margin recipe.
            if margin <= 0.0 or rt is None or g in covered_goods:
                recipe_targets[rname] = (0, 0.0)
                continue
            covered_goods.add(g)

            if rt.is_hard_capped:
                target_u = max(0, rt.inferred_prod_cap - curr_fin)
                if remaining_cap is not None:
                    target_u = min(target_u, remaining_cap)
                    remaining_cap -= target_u
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
                crowding = float(crowding_map.get(g, 0.0))
                if is_blitz and g in blitz_top_goods and unsold == 0:
                    if supply_per_team < 1.8:
                        # Hyper-scarce lottery rooms (<1.8 units/team): don't over-order unmatched parts.
                        desired_total = max(1, min(2, int(ceil(vel * 1.1 + 0.5))))
                    else:
                        # Calibrated early blitz: build strong monopoly batch on top-2 goods.
                        blitz_target = max(4, min(6, rt.inferred_prod_cap + 3))
                        desired_total = max(blitz_target, int(ceil(vel * 1.8 + 2.0)))
                elif is_blitz and num_finished > 2 and g not in blitz_top_goods:
                    # During blitz, focus budget on top-2 high-tier goods; skip crowded low-tier.
                    desired_total = curr_fin
                elif is_crowded and not is_blitz and not is_expansion and crowding > 0.45 and (curr_fin > 0 or unsold >= 1):
                    # Red-ocean avoidance: stop buying materials for goods flooded by rivals.
                    desired_total = curr_fin
                elif total_sold == 0:
                    if unsold >= 2 or curr_fin >= max(3, rt.inferred_prod_cap):
                        desired_total = curr_fin
                    else:
                        desired_total = max(2, min(10, rt.inferred_prod_cap))
                elif unsold >= 2:
                    desired_total = max(0, int(vel * 0.6))
                elif unsold == 1:
                    desired_total = max(1, int(ceil(vel * 0.95)))
                elif stockouts >= 1:
                    ramp_mult = 2.25 if is_expansion else 1.65
                    desired_total = max(
                        int(ceil(vel * ramp_mult + (4.0 if is_expansion else 1.5))),
                        int(ceil(last_sold * (1.75 if is_expansion else 1.4) + 1.0)),
                    )
                else:
                    flow_mult = 1.65 if is_expansion else 1.30
                    desired_total = max(1, int(ceil(vel * flow_mult + (2.0 if is_expansion else 0.5))))

            # Endgame taper: avoid aggressive cutoffs at uncertain early checkpoints past turn 40,
            # while still tapering at the true end of each checkpoint window.
            taper_window = 2 if state.turn > 40 else 4
            if remaining_turns <= taper_window:
                eff_vel = max(0.8, vel) if total_sold > 0 else 0.5
                if remaining_turns == 1:
                    endgame_max = int(eff_vel * 0.85) if curr_fin == 0 else 0
                elif remaining_turns == 2:
                    endgame_max = int(ceil(eff_vel * 1.55))
                else:
                    endgame_max = int(ceil(eff_vel * (remaining_turns - 0.5)))
                desired_total = min(desired_total, endgame_max)

            net_u = max(0, desired_total - curr_fin)
            if net_u > 0 and recipe.inputs:
                # Never target more recipe sets than existing raw + total lot supply can support.
                phys_cap = min(
                    (max(0, int(state.inventory.get(m, 0))) + lot_qty_map.get(m, 0))
                    // max(1, int(amt))
                    for m, amt in recipe.inputs.items()
                )
                net_u = min(net_u, max(0, phys_cap))

                # In crowded markets, condition complementary purchases on bottleneck fill rates
                # so we don't accumulate unmatched secondary materials when the bottleneck stalls.
                if is_crowded and state.turn >= 2 and net_u > 0:
                    achievable = net_u
                    for m, amt in recipe.inputs.items():
                        amt_i = max(1, int(amt))
                        curr_m = max(0, int(state.inventory.get(m, 0)))
                        lot_m = lot_qty_map.get(m, 0)
                        ms_m = self._materials.get(m)
                        fill_m = ms_m.ema_fill_rate if ms_m is not None else 0.65
                        if fill_m < 0.38:
                            est_won = int(ceil(lot_m * min(1.0, fill_m * 1.45)))
                            if supply_per_team >= 1.8 and lot_m >= amt_i and fill_m >= 0.15:
                                est_won = max(amt_i, est_won)
                            achievable = min(achievable, (curr_m + est_won) // amt_i)
                    net_u = min(net_u, max(0, achievable))

                # Ultra-scarce rooms (<1.6 total lot units per team): only pursue recipes where
                # at most 2 distinct input materials are missing, avoiding 4-way partial-fill losses.
                if is_crowded and supply_per_team < 1.6 and net_u > 0:
                    missing_mats = sum(
                        1 for m, amt in recipe.inputs.items()
                        if max(0, int(state.inventory.get(m, 0))) < max(1, int(amt))
                    )
                    if missing_mats > 2:
                        net_u = 0

            if remaining_cap is not None:
                net_u = min(net_u, remaining_cap)
                remaining_cap -= net_u
            recipe_targets[rname] = (net_u, margin)

        return recipe_targets, max_material_wtp, est_prices

    def _estimate_all_good_prices(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> dict[str, float]:
        """Return estimated sale price for every finished good (cheapest recipe per good)."""
        raw_costs: dict[str, float] = {}
        init_costs: dict[str, float] = {}
        input_counts: dict[str, int] = {}
        for recipe in goods_config.recipes.values():
            g = recipe.output_good
            c = sum(
                self._materials[mat].init_reserve * int(amt)
                for mat, amt in recipe.inputs.items()
                if mat in self._materials
            )
            u = max(1, sum(int(amt) for amt in recipe.inputs.values()))
            if g not in raw_costs or c < raw_costs[g] or (abs(c - raw_costs[g]) < 1e-6 and u < input_counts[g]):
                raw_costs[g] = max(0.0, c)
                init_costs[g] = max(0.01, c)
                input_counts[g] = u

        c_min = min(init_costs.values()) if init_costs else 1.0
        c_max = max(init_costs.values()) if init_costs else 1.0
        max_raw_cost = max(raw_costs.values()) if raw_costs else 0.0
        u_min = min(input_counts.values()) if input_counts else 1
        u_max = max(input_counts.values()) if input_counts else 1
        num_finished = len(goods_config.finished_goods)

        est: dict[str, float] = {}
        nominal_materials = (
            (max_raw_cost <= 0.05 and self._starting_capital >= 500.0)
            or (
                max_raw_cost <= 3.0
                and self._starting_capital >= 1000.0
                and self._total_init_lots >= 300
            )
        )
        for g in goods_config.finished_goods:
            icost = init_costs.get(g, 25.0)
            iunits = input_counts.get(g, 2)
            r_units = (iunits - u_min) / max(1e-6, float(u_max - u_min))
            if nominal_materials:
                # Free / near-zero reserve materials (e.g. tutorial_pricing, priv_zero_reserve).
                base_prior = max(icost * 5.0, self._starting_capital * 0.025)
                prior = base_prior * (0.85 + 1.3 * r_units) if num_finished > 1 else base_prior
            elif num_finished > 1 and (c_max - c_min > 1e-6 or u_max > u_min):
                r_cost = (icost - c_min) / max(1e-6, c_max - c_min)
                r_tier = 0.72 * r_cost + 0.28 * r_units
                prior = icost * (5.0 + 10.0 * (r_tier ** 1.2))
            else:
                prior = icost * 5.0

            observed = ctx.est_sale_price.get(g, 0.0)
            est[g] = observed if observed > icost * 1.2 else prior
        return est

    # ------------------------------------------------------------------
    # 3-Regime Pay-Your-Bid Clearing Price Model
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_one_batch_need(goods_config: GoodsConfig) -> dict[str, int]:
        """Baseline batch demand per material taking max across alternative recipes per good."""
        per_good: dict[str, dict[str, int]] = {}
        for recipe in goods_config.recipes.values():
            g = recipe.output_good
            g_map = per_good.setdefault(g, {})
            for m, amt in recipe.inputs.items():
                amt_i = int(amt)
                if amt_i > g_map.get(m, 0):
                    g_map[m] = amt_i
        totals: dict[str, int] = {m: 0 for m in goods_config.raw_materials}
        for g_map in per_good.values():
            for m, amt_i in g_map.items():
                totals[m] = totals.get(m, 0) + amt_i
        return totals

    def _enforce_bundle_roi_caps(
        self,
        planned_bids: dict[str, tuple[int, float, int, float]],
        recipe_targets: dict[str, tuple[int, float]],
        est_prices: dict[str, float],
        lot_reserves: dict[str, float],
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> dict[str, tuple[int, float, int, float]]:
        """Scale down bids if combined recipe input bids violate safe output ROI."""
        if not planned_bids:
            return planned_bids

        active_recipes = [
            (rname, goods_config.recipes[rname])
            for rname, (qty, _m) in recipe_targets.items()
            if qty > 0 and rname in goods_config.recipes
        ]
        if not active_recipes:
            return planned_bids

        mat_max_allowed: dict[str, float] = {}
        for rname, recipe in active_recipes:
            sale_p = est_prices.get(recipe.output_good, 0.0)
            if sale_p <= 0.0:
                continue
            max_allowed_bundle = sale_p * 0.88
            bid_cost = 0.0
            other_cost = 0.0
            for m, amt in recipe.inputs.items():
                amt_i = max(1, int(amt))
                if m in planned_bids:
                    bid_cost += planned_bids[m][1] * amt_i
                else:
                    other_cost += lot_reserves.get(m, ctx.avg_material_cost.get(m, 0.0)) * amt_i
            if bid_cost <= 1e-9:
                continue
            if bid_cost + other_cost <= max_allowed_bundle:
                scale = 1.0
            else:
                avail_for_bids = max(0.0, max_allowed_bundle - other_cost)
                scale = avail_for_bids / bid_cost
            for m in recipe.inputs:
                if m in planned_bids:
                    allowed_p = planned_bids[m][1] * scale
                    if m not in mat_max_allowed or allowed_p > mat_max_allowed[m]:
                        mat_max_allowed[m] = allowed_p

        filtered: dict[str, tuple[int, float, int, float]] = {}
        for m, (qty, price, regime, prio) in planned_bids.items():
            cap = mat_max_allowed.get(m)
            if cap is not None and price > cap:
                rprice = lot_reserves.get(m, 0.0)
                if cap < rprice:
                    continue
                price = max(rprice * 1.0005 if rprice > 1e-6 else 0.01, cap)
            filtered[m] = (qty, price, regime, prio)
        return filtered

    def _compute_optimal_bid(
        self,
        lot,
        net_qty: int,
        max_wtp: float,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
        remaining_turns: int,
        one_batch_need: int,
        one_batch_total_cost: float,
        lot_reserves: dict[str, float],
    ) -> tuple[float, int, int]:
        """Select optimal `(bid_price, regime, adj_qty)` for `lot`."""
        mat = lot.good
        rprice = float(lot.reserve_price)
        lot_qty = int(lot.quantity)
        ms = self._materials[mat]

        d_high, d_mid = self._estimate_opponent_demand(
            mat, rprice, state, ctx, one_batch_need, one_batch_total_cost
        )
        d_high_scaled = int(ceil(d_high * ms.demand_scale))
        d_mid_scaled = int(ceil(d_mid * ms.demand_scale))

        free_after_all = max(0, lot_qty - d_high_scaled - d_mid_scaled)
        free_after_high = max(0, lot_qty - d_high_scaled)

        # When reserve_price is 0.0 (e.g. priv_zero_reserve), bid along `ms.zero_bid_price`
        # rather than compounding `avg_material_cost * high_mult` recursively.
        zero_reserve = rprice <= 1e-6

        # If the uncontested surplus in the lot already satisfies most of our need
        # (or in a single-good auction where uncontested units are abundant),
        # cap quantity to `free_after_all` and win 100% at the reserve price!
        if free_after_all >= net_qty:
            regime = 1
            raw_price = 0.05 if zero_reserve else rprice * 1.0012
            adj_qty = net_qty
        elif (
            len(goods_config.finished_goods) == 1
            and free_after_all >= max(2, int(net_qty * 0.65))
        ):
            regime = 1
            raw_price = 0.05 if zero_reserve else rprice * 1.0012
            adj_qty = free_after_all
        elif free_after_high >= net_qty:
            regime = 2
            raw_price = (
                max(0.12, min(max_wtp * 0.35, ms.zero_bid_price * 0.90))
                if zero_reserve
                else rprice * ms.mid_mult
            )
            adj_qty = net_qty
        else:
            regime = 3
            if zero_reserve:
                raw_price = max(0.20, min(max_wtp * 0.45, ms.zero_bid_price))
            elif getattr(ctx, "is_blitz_window", False):
                blitz_mult = max(ms.high_mult, 2.25)
                raw_price = rprice * blitz_mult
            else:
                raw_price = rprice * ms.high_mult
            adj_qty = net_qty

        # On the final turn, avoid paying high-contest premiums.
        if remaining_turns == 1 and regime == 3 and len(goods_config.finished_goods) > 1:
            if free_after_high > 0:
                regime = 2
                raw_price = (
                    max(0.12, ms.zero_bid_price * 0.90)
                    if zero_reserve
                    else rprice * ms.mid_mult
                )
                adj_qty = min(net_qty, free_after_high)
            else:
                return -1.0, regime, 0

        eff_wtp_cap = (
            max_wtp * 0.55
            if getattr(ctx, "is_blitz_window", False) and regime == 3
            else max_wtp
        )
        capped_price = min(raw_price, eff_wtp_cap)
        if capped_price < rprice:
            return -1.0, regime, 0
        min_valid = 0.01 if zero_reserve else rprice * 1.0005
        return max(min_valid, capped_price), regime, adj_qty

    def _estimate_opponent_demand(
        self,
        mat: str,
        rprice: float,
        state: Observation,
        ctx: MarketContext,
        one_batch_need: int,
        one_batch_total_cost: float,
    ) -> tuple[int, int]:
        """Estimate `(d_high, d_mid)` opponent demand for `mat` from public state."""
        if one_batch_need <= 0:
            return 0, 0

        turn = state.turn
        active_opps = self._active_opponents
        own_id = ctx.team_id
        pub_cap = state.public_capital
        opp_ids = [
            tid
            for tid in state.public_inventories
            if (own_id is None or tid != own_id)
            and (turn == 1 or tid in active_opps)
            and pub_cap.get(tid, 0.0) > rprice
        ]
        if not opp_ids:
            return 0, 0

        demands: list[tuple[int, int]] = []
        planned_4x_cost = 4.0 * one_batch_total_cost * 1.45
        pub_invs = state.public_inventories
        for tid in opp_ids:
            inv_m = max(0, int(pub_invs.get(tid, {}).get(mat, 0)))
            cap = max(0.0, float(pub_cap.get(tid, 0.0)))

            mid_need = max(0, one_batch_need - inv_m)
            raw_high_need = max(0, 4 * one_batch_need - inv_m)
            if one_batch_total_cost > 0:
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
        ctx: MarketContext,
        remaining_turns: int,
    ) -> dict[str, tuple[int, float, int]]:
        """Ensure total auction commitment stays safely within `state.capital`."""
        if not planned_bids:
            return {}

        is_crowded = bool(getattr(ctx, "is_crowded_tournament", False))
        is_expansion = bool(getattr(ctx, "is_high_volume_expansion", False))
        is_blitz = bool(getattr(ctx, "is_blitz_window", False))
        if is_blitz and state.turn <= 4:
            budget_share = 0.85
        elif is_expansion and remaining_turns > 3:
            budget_share = 0.82
        elif remaining_turns > 2:
            budget_share = 0.78
        elif remaining_turns == 2:
            budget_share = 0.50
        else:
            budget_share = 0.30

        # Post-blitz stop-loss in crowded tournament rooms: protect cash when shelf
        # inventory grows faster than sales velocity.
        if is_crowded and not is_blitz and remaining_turns > 2:
            total_fin_inv = sum(
                max(0, int(state.inventory.get(g, 0)))
                for g in goods_config.finished_goods
            )
            total_vel = sum(
                float(getattr(ctx, "own_sold_ema", {}).get(g, 0.0))
                for g in goods_config.finished_goods
            )
            if total_fin_inv > max(5, int(ceil(2.2 * total_vel))):
                budget_share = min(budget_share, 0.35)
            elif total_fin_inv > max(3, int(ceil(1.5 * total_vel))):
                budget_share = min(budget_share, 0.52)

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
        # First pass: proportional floored allocation so complementary materials aren't starved.
        for mat, (qty, price, regime, _prio) in ordered:
            if price <= 0:
                scaled[mat] = (qty, price, regime)
                continue
            rem_budget = max_budget - running_spend
            max_affordable = int(rem_budget // price)
            if max_affordable <= 0:
                continue
            target_q = int(qty * scale)
            final_q = min(qty, target_q, max_affordable)
            if final_q > 0:
                scaled[mat] = (final_q, price, regime)
                running_spend += final_q * price

        # Second pass: spend leftover budget (+1 unit at a time in priority order).
        for mat, (qty, price, regime, _prio) in ordered:
            if price <= 0:
                continue
            cur_q = scaled[mat][0] if mat in scaled else 0
            if cur_q >= qty:
                continue
            rem_budget = max_budget - running_spend
            if rem_budget >= price:
                scaled[mat] = (cur_q + 1, price, regime)
                running_spend += price

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
