"""Phase 2: Demand-aware, margin-maximizing production strategy (Owner: Persona 3).

Converts raw materials into finished goods using:
1. Diminishing marginal utility per finished good (demand saturation, cold-start
   exploration bonus, competitor shelf scarcity).
2. Resource-weighted integer greedy allocation + bounded local swap search
   (handles multi-material bottlenecks and optional `capacity_per_turn`).
3. Two-wave staged `AssemblyOrder` output so materials won in Phase 1 of the
   *same* turn are assembled immediately without a 1-turn delay.
"""

from __future__ import annotations

from math import ceil
from time import perf_counter

from dme_simulator.agents import AssemblyOrder, Observation
from dme_simulator.config import GoodsConfig

from .state import MarketContext


class ProductionStrategy:
    """Computes ordered assembly instructions for raw materials -> finished goods."""

    _MAX_GREEDY_STEPS = 80
    _MAX_SWAP_ITERS = 30
    _MAX_LOCAL_SEARCH_SEC = 0.012

    def __init__(self) -> None:
        pass

    def decide_assembly(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> list[AssemblyOrder]:
        """Return staged `AssemblyOrder` list optimized for guaranteed + same-turn won stock."""
        recipes = goods_config.recipes
        if not recipes:
            return []

        lot_by_good = {lot.good: lot for lot in state.auction_lots}

        # Aggregate valid planned bids (price >= reserve) per raw material.
        valid_bid_qty: dict[str, int] = {}
        for bid in ctx.planned_bids:
            lot = lot_by_good.get(bid.good)
            if lot is None or bid.quantity <= 0:
                continue
            if bid.price_per_unit >= lot.reserve_price - 1e-9:
                valid_bid_qty[bid.good] = min(
                    lot.quantity,
                    valid_bid_qty.get(bid.good, 0) + bid.quantity,
                )

        # Build two raw-material pools:
        # - Wave 1 (projected): current stock + expected fill from this turn's bids.
        # - Wave 2 (optimistic): current stock + 100% fill of valid bids.
        projected_raw: dict[str, int] = {}
        optimistic_raw: dict[str, int] = {}
        for mat in goods_config.raw_materials:
            owned = state.inventory.get(mat, 0)
            bid_q = valid_bid_qty.get(mat, 0)
            fill = ctx.auction_fill_rate.get(mat, 0.60)
            expected_add = int(round(bid_q * fill))
            if bid_q > 0 and expected_add == 0 and fill >= 0.25:
                expected_add = 1
            projected_raw[mat] = owned + expected_add
            optimistic_raw[mat] = owned + bid_q

        # Compute per-good soft shelf-stock targets and economic parameters.
        target_stock = self._compute_target_stocks(state, goods_config, ctx)
        recipe_rev = {
            r_name: ctx.estimate_good_revenue(recipe.output_good, goods_config)
            for r_name, recipe in recipes.items()
        }
        recipe_cost = {
            r_name: ctx.estimate_recipe_cost(r_name, goods_config)
            for r_name in recipes
        }
        # Base finished-good inventory already on shelf before Phase 2.
        shelf_inv = {
            g: state.inventory.get(g, 0) for g in goods_config.finished_goods
        }
        mat_weights = self._compute_material_weights(
            goods_config, ctx, projected_raw, target_stock, shelf_inv, recipe_rev, recipe_cost
        )
        cap = ctx.estimated_capacity_per_turn

        # Wave 1: optimize over projected post-auction materials.
        wave1_plan, wave1_sequence = self._solve_production_plan(
            raw_pool=projected_raw,
            shelf_inv=shelf_inv,
            goods_config=goods_config,
            ctx=ctx,
            target_stock=target_stock,
            recipe_rev=recipe_rev,
            recipe_cost=recipe_cost,
            mat_weights=mat_weights,
            capacity_limit=cap,
        )

        # Wave 2: compute upside overflow if Phase 1 bids fill better than expected.
        wave2_sequence: list[tuple[str, int]] = []
        has_upside = any(
            optimistic_raw[m] > projected_raw[m] for m in goods_config.raw_materials
        )
        if has_upside:
            rem_raw = dict(optimistic_raw)
            after_w1_shelf = dict(shelf_inv)
            used_cap = 0
            for r_name, qty in wave1_plan.items():
                if qty <= 0:
                    continue
                recipe = recipes[r_name]
                for mat, units in recipe.inputs.items():
                    rem_raw[mat] = max(0, rem_raw.get(mat, 0) - units * qty)
                after_w1_shelf[recipe.output_good] = (
                    after_w1_shelf.get(recipe.output_good, 0) + qty
                )
                used_cap += qty

            rem_cap = None if cap is None else max(0, cap - used_cap)
            if rem_cap is None or rem_cap > 0:
                _, wave2_sequence = self._solve_production_plan(
                    raw_pool=rem_raw,
                    shelf_inv=after_w1_shelf,
                    goods_config=goods_config,
                    ctx=ctx,
                    target_stock=target_stock,
                    recipe_rev=recipe_rev,
                    recipe_cost=recipe_cost,
                    mat_weights=mat_weights,
                    capacity_limit=rem_cap,
                )

        # Publish desired raw-material demand & marginal shadow values for coordination.
        self._update_target_material_demand(
            goods_config, ctx, target_stock, shelf_inv, recipe_rev, recipe_cost
        )
        self._update_shadow_values(
            projected_raw=projected_raw,
            wave1_plan=wave1_plan,
            shelf_inv=shelf_inv,
            goods_config=goods_config,
            ctx=ctx,
            target_stock=target_stock,
            recipe_rev=recipe_rev,
            recipe_cost=recipe_cost,
        )

        return self._coalesce_orders(wave1_sequence + wave2_sequence)

    def _compute_target_stocks(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> dict[str, int]:
        """Estimate healthy finished-good shelf targets scaled to market size & velocity."""
        recipes = goods_config.recipes
        avg_inputs = (
            sum(sum(r.inputs.values()) for r in recipes.values()) / max(1, len(recipes))
        )
        total_lot_units = sum(lot.quantity for lot in state.auction_lots)
        team_count = max(1, ctx.active_competitor_count + 1)
        num_goods = max(1, len(goods_config.finished_goods))

        # Market-scale baseline per finished good per team (adapts to default, scaled, and 240-turn).
        est_market_good_units = total_lot_units / max(1.0, avg_inputs * team_count * num_goods)
        probe_cap = 250 if getattr(ctx, "is_high_volume_expansion", False) else 40
        market_probe = max(2, min(probe_cap, ceil(est_market_good_units * 1.4)))

        # During a crowded-room blitz (Turns 1-6), identify the top 2 highest-margin goods
        # so blitz-won materials immediately convert into sellable monopoly units.
        blitz_top_goods: set[str] = set()
        if getattr(ctx, "is_blitz_window", False) and num_goods > 1:
            good_best_margin: dict[str, float] = {}
            for r_name, recipe in recipes.items():
                m_val = ctx.recipe_margins.get(r_name, 0.0)
                g = recipe.output_good
                if g not in good_best_margin or m_val > good_best_margin[g]:
                    good_best_margin[g] = m_val
            ranked_g = sorted(good_best_margin, key=lambda g: -good_best_margin[g])
            blitz_top_goods = set(ranked_g[: min(2, len(ranked_g))])

        targets: dict[str, int] = {}
        for good in goods_config.finished_goods:
            sold_count = ctx.sales_count_by_good.get(good, 0)
            vel = ctx.own_sold_ema.get(good, 0.0)
            last_sold = ctx.own_sold_last_turn.get(good, 0)
            stockouts = ctx.own_stockout_streak.get(good, 0)
            unsold = ctx.own_unsold_streak.get(good, 0)

            if sold_count == 0:
                # Untested good: keep a modest probe target unless stock is already sitting unsold.
                if unsold >= 2:
                    targets[good] = 0
                elif unsold == 1:
                    targets[good] = 1
                elif good in blitz_top_goods:
                    targets[good] = max(5, min(market_probe + 3, 8))
                else:
                    targets[good] = max(1, min(market_probe, 4))
            elif stockouts >= 1:
                # Selling out: ramp up capacity rapidly to capture unmet demand.
                ramp = max(vel * (1.8 + 0.3 * min(stockouts, 3)) + 2.0, last_sold * 1.5 + 1.0)
                targets[good] = max(market_probe, ceil(ramp))
            elif unsold >= 2:
                # Sitting unsold for multiple turns: stop building until shelf clears.
                targets[good] = max(0, int(vel * 0.5))
            elif unsold == 1:
                # Single unsold turn: hold steady around velocity without over-building.
                targets[good] = max(1, ceil(vel * 1.0))
            else:
                base_t = max(market_probe, ceil(vel * 1.45 + 1.0))
                targets[good] = max(base_t, 6) if good in blitz_top_goods else base_t
        return targets

    def _compute_material_weights(
        self,
        goods_config: GoodsConfig,
        ctx: MarketContext,
        raw_pool: dict[str, int],
        target_stock: dict[str, int],
        shelf_inv: dict[str, int],
        recipe_rev: dict[str, float],
        recipe_cost: dict[str, float],
    ) -> dict[str, float]:
        """Compute scarcity-adjusted resource weights with surplus flushing."""
        demand_weight: dict[str, float] = {m: 0.0 for m in goods_config.raw_materials}
        est_mat_need: dict[str, int] = {m: 0 for m in goods_config.raw_materials}
        for r_name, recipe in goods_config.recipes.items():
            margin = max(1.0, recipe_rev[r_name] - recipe_cost[r_name])
            needed_goods = max(1, target_stock.get(recipe.output_good, 2) - shelf_inv.get(recipe.output_good, 0))
            for mat, units in recipe.inputs.items():
                demand_weight[mat] = demand_weight.get(mat, 0.0) + units * margin
                est_mat_need[mat] = est_mat_need.get(mat, 0) + units * needed_goods

        avg_cost = (
            sum(ctx.avg_material_cost.get(m, 0.0) for m in goods_config.raw_materials)
            / max(1, len(goods_config.raw_materials))
        )
        base_floor = max(1.0, avg_cost * 0.25)

        weights: dict[str, float] = {}
        for mat in goods_config.raw_materials:
            unit_cost = max(base_floor, ctx.avg_material_cost.get(mat, 0.0))
            scarcity = ctx.lot_scarcity.get(mat, 0.5)
            pool_qty = raw_pool.get(mat, 0)
            pool_tightness = 1.0 / max(1.0, float(pool_qty))
            demand_boost = 0.25 if demand_weight.get(mat, 0.0) > 0.0 else 0.0
            weight = unit_cost * (1.0 + 0.6 * scarcity + demand_boost + 0.5 * pool_tightness)

            # Surplus flushing: if we hold far more of a material than recipes need,
            # discount its weight so recipes burning abundant surplus aren't penalized.
            need = max(1, est_mat_need.get(mat, 1))
            if pool_qty > 2.5 * need:
                surplus_ratio = need / float(pool_qty)
                weight *= max(0.15, surplus_ratio * 2.5)
            weights[mat] = max(0.15 * base_floor, weight)
        return weights

    def _marginal_utility(
        self,
        recipe_name: str,
        current_good_qty: int,
        goods_config: GoodsConfig,
        ctx: MarketContext,
        target_stock: dict[str, int],
        recipe_rev: dict[str, float],
        recipe_cost: dict[str, float],
    ) -> float:
        """Diminishing marginal value of producing one additional unit of `recipe_name`."""
        good = goods_config.recipes[recipe_name].output_good
        rev = recipe_rev[recipe_name]
        cost = recipe_cost[recipe_name]
        if rev <= 0.0:
            return 0.0

        # Raw materials already held are mostly sunk cost; apply a light opportunity-cost
        # discount (0.25x replacement cost) so higher-margin goods win without stranding raw stock.
        base_value = rev - 0.25 * cost
        if base_value <= 0.0:
            # Even if observed price < cost, allow low-priority recovery only if rev > 0.
            base_value = rev * 0.15

        # Penalize recipes whose realized market price is clearly below replacement cost.
        if ctx.sales_count_by_good.get(good, 0) >= 2 and rev < cost * 0.95:
            base_value *= 0.25

        # Cold-start exploration bonus for the first probe unit of untested goods.
        unsold_streak = ctx.own_unsold_streak.get(good, 0)
        if ctx.sales_count_by_good.get(good, 0) == 0 and current_good_qty == 0 and unsold_streak == 0:
            base_value *= 1.35

        # Penalize goods sitting unsold on our shelf so we don't keep piling up dead stock.
        if unsold_streak >= 1:
            base_value *= 0.5 ** min(unsold_streak, 4)

        # Crowding-aware niche routing: reward open niches and penalize crowded graveyards.
        if ctx.active_competitor_count > 0:
            sellers = ctx.competitor_sellers_by_good.get(
                good, 0 if ctx.competitor_finished_inventory.get(good, 0) == 0 else 1
            )
            crowding = ctx.crowding_ratio_by_good.get(good, 0.0)
            if ctx.active_competitor_count >= 8:
                if sellers == 0:
                    base_value *= 1.30
                elif sellers == 1:
                    base_value *= 1.15
                elif crowding >= 0.40:
                    base_value *= max(0.45, 1.0 - 0.50 * crowding)
            elif ctx.competitor_finished_inventory.get(good, 0) == 0:
                base_value *= 1.15

        # Diminishing returns beyond target_stock (prevents piling up saturated goods).
        target = target_stock.get(good, 2)
        if current_good_qty >= target:
            excess = current_good_qty - target + 1
            if unsold_streak >= 1:
                soft_ceiling = target + 1
            else:
                soft_ceiling = max(target * 2 + 1, target + 3)
            if current_good_qty >= soft_ceiling:
                return 0.0
            base_value *= 0.40 ** excess

        return base_value

    def _solve_production_plan(
        self,
        raw_pool: dict[str, int],
        shelf_inv: dict[str, int],
        goods_config: GoodsConfig,
        ctx: MarketContext,
        target_stock: dict[str, int],
        recipe_rev: dict[str, float],
        recipe_cost: dict[str, float],
        mat_weights: dict[str, float],
        capacity_limit: int | None,
    ) -> tuple[dict[str, int], list[tuple[str, int]]]:
        """Solve integer production allocation via resource-weighted greedy + local swap."""
        recipes = goods_config.recipes
        rem_raw = dict(raw_pool)
        planned_good = dict(shelf_inv)
        plan: dict[str, int] = {r_name: 0 for r_name in recipes}
        sequence: list[tuple[str, int]] = []

        # Compute rough upper bound on buildable units to choose chunk size for speed (<1 ms).
        raw_max_possible = 0
        for recipe in recipes.values():
            if not recipe.inputs:
                continue
            can_make = min(rem_raw.get(m, 0) // u for m, u in recipe.inputs.items())
            raw_max_possible = max(raw_max_possible, can_make)
        max_possible = (
            min(raw_max_possible, capacity_limit)
            if capacity_limit is not None
            else raw_max_possible
        )
        if max_possible <= 0:
            return plan, sequence

        step_size = max(1, max_possible // 60)

        # Precompute resource denominator per recipe.
        # When capacity_per_turn is binding, shift denominator toward 1.0 (unit margin).
        cap_binding = capacity_limit is not None and capacity_limit < raw_max_possible
        avg_recipe_weight = (
            sum(
                sum(mat_weights.get(m, 1.0) * u for m, u in r.inputs.items())
                for r in recipes.values()
            )
            / max(1, len(recipes))
        )
        recipe_denom: dict[str, float] = {}
        for r_name, recipe in recipes.items():
            mat_w = sum(mat_weights.get(m, 1.0) * u for m, u in recipe.inputs.items())
            if cap_binding:
                recipe_denom[r_name] = 0.75 * avg_recipe_weight + 0.25 * max(1.0, mat_w)
            else:
                recipe_denom[r_name] = max(1.0, mat_w)

        rem_cap = capacity_limit
        steps = 0

        # Phase A: Greedy allocation by marginal utility per weighted resource cost.
        while steps < self._MAX_GREEDY_STEPS:
            if rem_cap is not None and rem_cap <= 0:
                break

            best_recipe: str | None = None
            best_score = 0.0
            best_chunk = 0

            for r_name, recipe in recipes.items():
                if not recipe.inputs:
                    continue
                can_make = min(rem_raw.get(m, 0) // u for m, u in recipe.inputs.items())
                if rem_cap is not None:
                    can_make = min(can_make, rem_cap)
                if can_make <= 0:
                    continue

                good = recipe.output_good
                util = self._marginal_utility(
                    r_name,
                    planned_good.get(good, 0),
                    goods_config,
                    ctx,
                    target_stock,
                    recipe_rev,
                    recipe_cost,
                )
                if util <= 0.0:
                    continue

                score = util / recipe_denom[r_name]
                if score > best_score:
                    best_score = score
                    best_recipe = r_name
                    # Keep step=1 near/below target_stock for fine interleaving.
                    dist_to_target = max(1, target_stock.get(good, 2) - planned_good.get(good, 0))
                    best_chunk = min(can_make, step_size, dist_to_target)

            if best_recipe is None or best_chunk <= 0:
                break

            recipe = recipes[best_recipe]
            for mat, units in recipe.inputs.items():
                rem_raw[mat] -= units * best_chunk
            planned_good[recipe.output_good] = (
                planned_good.get(recipe.output_good, 0) + best_chunk
            )
            plan[best_recipe] += best_chunk
            sequence.append((best_recipe, best_chunk))
            if rem_cap is not None:
                rem_cap -= best_chunk
            steps += 1

        # Phase B: Bounded 1-for-k local swap improvement on integer plan.
        improved = self._local_swap_improve(
            plan=plan,
            rem_raw=rem_raw,
            planned_good=planned_good,
            rem_cap=rem_cap,
            goods_config=goods_config,
            ctx=ctx,
            target_stock=target_stock,
            recipe_rev=recipe_rev,
            recipe_cost=recipe_cost,
            recipe_denom=recipe_denom,
        )
        if improved:
            sequence = self._rebuild_priority_sequence(
                plan=plan,
                shelf_inv=shelf_inv,
                goods_config=goods_config,
                ctx=ctx,
                target_stock=target_stock,
                recipe_rev=recipe_rev,
                recipe_cost=recipe_cost,
                recipe_denom=recipe_denom,
            )

        return plan, sequence

    def _local_swap_improve(
        self,
        plan: dict[str, int],
        rem_raw: dict[str, int],
        planned_good: dict[str, int],
        rem_cap: int | None,
        goods_config: GoodsConfig,
        ctx: MarketContext,
        target_stock: dict[str, int],
        recipe_rev: dict[str, float],
        recipe_cost: dict[str, float],
        recipe_denom: dict[str, float],
    ) -> bool:
        """Try removing 1 unit of a planned recipe to fit strictly higher total utility."""
        recipes = goods_config.recipes
        if len(recipes) <= 1:
            return False

        any_improved = False
        deadline = perf_counter() + self._MAX_LOCAL_SEARCH_SEC
        for _ in range(self._MAX_SWAP_ITERS):
            if perf_counter() >= deadline:
                break
            best_gain = 1e-6
            best_move: tuple[str, list[str]] | None = None

            active_recipes = [r for r, qty in plan.items() if qty > 0]
            for remove_r in active_recipes:
                rem_recipe = recipes[remove_r]
                rem_good = rem_recipe.output_good
                # Utility lost by dropping the last unit of remove_r.
                lost_util = self._marginal_utility(
                    remove_r,
                    planned_good[rem_good] - 1,
                    goods_config,
                    ctx,
                    target_stock,
                    recipe_rev,
                    recipe_cost,
                )

                # Temporarily free remove_r's materials and capacity.
                trial_raw = dict(rem_raw)
                for mat, units in rem_recipe.inputs.items():
                    trial_raw[mat] = trial_raw.get(mat, 0) + units
                trial_good = dict(planned_good)
                trial_good[rem_good] -= 1
                trial_cap = None if rem_cap is None else rem_cap + 1

                # Greedily add up to 2 replacement units from other recipes.
                added: list[str] = []
                gained_util = 0.0
                for _ in range(2):
                    if trial_cap is not None and trial_cap <= 0:
                        break
                    cand_r: str | None = None
                    cand_score = 0.0
                    cand_util = 0.0
                    for r_name, recipe in recipes.items():
                        if r_name == remove_r or not recipe.inputs:
                            continue
                        if any(trial_raw.get(m, 0) < u for m, u in recipe.inputs.items()):
                            continue
                        u_val = self._marginal_utility(
                            r_name,
                            trial_good.get(recipe.output_good, 0),
                            goods_config,
                            ctx,
                            target_stock,
                            recipe_rev,
                            recipe_cost,
                        )
                        if u_val <= 0.0:
                            continue
                        s_val = u_val / recipe_denom[r_name]
                        if s_val > cand_score:
                            cand_score = s_val
                            cand_r = r_name
                            cand_util = u_val
                    if cand_r is None:
                        break
                    added.append(cand_r)
                    gained_util += cand_util
                    c_rec = recipes[cand_r]
                    for mat, units in c_rec.inputs.items():
                        trial_raw[mat] -= units
                    trial_good[c_rec.output_good] = trial_good.get(c_rec.output_good, 0) + 1
                    if trial_cap is not None:
                        trial_cap -= 1

                net_gain = gained_util - lost_util
                if added and net_gain > best_gain:
                    best_gain = net_gain
                    best_move = (remove_r, added)

            if best_move is None:
                break

            remove_r, added = best_move
            rem_recipe = recipes[remove_r]
            plan[remove_r] -= 1
            for mat, units in rem_recipe.inputs.items():
                rem_raw[mat] = rem_raw.get(mat, 0) + units
            planned_good[rem_recipe.output_good] -= 1
            if rem_cap is not None:
                rem_cap += 1

            for add_r in added:
                add_recipe = recipes[add_r]
                plan[add_r] += 1
                for mat, units in add_recipe.inputs.items():
                    rem_raw[mat] -= units
                planned_good[add_recipe.output_good] = (
                    planned_good.get(add_recipe.output_good, 0) + 1
                )
                if rem_cap is not None:
                    rem_cap -= 1
            any_improved = True

        return any_improved

    def _rebuild_priority_sequence(
        self,
        plan: dict[str, int],
        shelf_inv: dict[str, int],
        goods_config: GoodsConfig,
        ctx: MarketContext,
        target_stock: dict[str, int],
        recipe_rev: dict[str, float],
        recipe_cost: dict[str, float],
        recipe_denom: dict[str, float],
    ) -> list[tuple[str, int]]:
        """Interleave finalized plan quantities in descending marginal-efficiency order."""
        remaining = {r: q for r, q in plan.items() if q > 0}
        sim_good = dict(shelf_inv)
        seq: list[tuple[str, int]] = []
        total_units = sum(remaining.values())
        chunk = max(1, total_units // 60)

        while remaining:
            best_r = max(
                remaining,
                key=lambda r: self._marginal_utility(
                    r,
                    sim_good.get(goods_config.recipes[r].output_good, 0),
                    goods_config,
                    ctx,
                    target_stock,
                    recipe_rev,
                    recipe_cost,
                )
                / recipe_denom[r],
            )
            take = min(chunk, remaining[best_r])
            seq.append((best_r, take))
            good = goods_config.recipes[best_r].output_good
            sim_good[good] = sim_good.get(good, 0) + take
            remaining[best_r] -= take
            if remaining[best_r] <= 0:
                del remaining[best_r]
        return seq

    def _update_target_material_demand(
        self,
        goods_config: GoodsConfig,
        ctx: MarketContext,
        target_stock: dict[str, int],
        shelf_inv: dict[str, int],
        recipe_rev: dict[str, float],
        recipe_cost: dict[str, float],
    ) -> None:
        """Populate `ctx.target_material_demand` for optional future bidding alignment."""
        # Pick best (highest margin) recipe per finished good.
        best_recipe_for_good: dict[str, str] = {}
        best_margin_for_good: dict[str, float] = {}
        for r_name, recipe in goods_config.recipes.items():
            margin = recipe_rev[r_name] - recipe_cost[r_name]
            good = recipe.output_good
            if good not in best_recipe_for_good or margin > best_margin_for_good[good]:
                best_recipe_for_good[good] = r_name
                best_margin_for_good[good] = margin

        demand: dict[str, int] = {m: 0 for m in goods_config.raw_materials}
        for good, target in target_stock.items():
            r_name = best_recipe_for_good.get(good)
            if r_name is None or best_margin_for_good.get(good, 0.0) <= 0.0:
                continue
            needed_goods = max(0, target - shelf_inv.get(good, 0))
            if needed_goods <= 0:
                continue
            for mat, units in goods_config.recipes[r_name].inputs.items():
                demand[mat] = demand.get(mat, 0) + units * needed_goods
        ctx.target_material_demand = demand

    def _update_shadow_values(
        self,
        projected_raw: dict[str, int],
        wave1_plan: dict[str, int],
        shelf_inv: dict[str, int],
        goods_config: GoodsConfig,
        ctx: MarketContext,
        target_stock: dict[str, int],
        recipe_rev: dict[str, float],
        recipe_cost: dict[str, float],
    ) -> None:
        """Estimate marginal revenue value (+1 unit) of each raw material."""
        recipes = goods_config.recipes
        rem_raw = dict(projected_raw)
        after_shelf = dict(shelf_inv)
        for r_name, qty in wave1_plan.items():
            if qty <= 0:
                continue
            recipe = recipes[r_name]
            for mat, units in recipe.inputs.items():
                rem_raw[mat] = max(0, rem_raw.get(mat, 0) - units * qty)
            after_shelf[recipe.output_good] = after_shelf.get(recipe.output_good, 0) + qty

        shadows: dict[str, float] = {}
        for mat in goods_config.raw_materials:
            baseline = ctx.material_shadow_value.get(mat, 0.0)
            best_unlock = 0.0
            for r_name, recipe in recipes.items():
                req_mat = recipe.inputs.get(mat, 0)
                if req_mat <= 0:
                    continue
                # Check if adding +1 unit of `mat` directly completes this recipe,
                # or reduces its missing-input deficit.
                missing_other = sum(
                    max(0, u - rem_raw.get(m, 0))
                    for m, u in recipe.inputs.items()
                    if m != mat
                )
                missing_self = max(0, req_mat - rem_raw.get(mat, 0))
                util = self._marginal_utility(
                    r_name,
                    after_shelf.get(recipe.output_good, 0),
                    goods_config,
                    ctx,
                    target_stock,
                    recipe_rev,
                    recipe_cost,
                )
                if util <= 0.0:
                    continue
                if missing_other == 0 and missing_self == 1:
                    # Direct bottleneck: +1 unit immediately unlocks an entire finished good!
                    best_unlock = max(best_unlock, util)
                elif missing_self > 0:
                    total_missing = missing_other + missing_self
                    best_unlock = max(best_unlock, util / max(1, total_missing))

            # Blend direct bottleneck unlock with baseline economic value per unit.
            if best_unlock > 0.0:
                shadows[mat] = 0.65 * best_unlock + 0.35 * baseline
            else:
                # Non-bottleneck surplus material: discount baseline value.
                shadows[mat] = 0.5 * baseline
        ctx.material_shadow_value = shadows

    @staticmethod
    def _coalesce_orders(sequence: list[tuple[str, int]]) -> list[AssemblyOrder]:
        """Merge consecutive identical recipe steps into compact `AssemblyOrder`s."""
        orders: list[AssemblyOrder] = []
        for r_name, qty in sequence:
            if qty <= 0:
                continue
            if orders and orders[-1].recipe_name == r_name:
                orders[-1] = AssemblyOrder(r_name, orders[-1].quantity + qty)
            else:
                orders.append(AssemblyOrder(r_name, qty))
        return orders
