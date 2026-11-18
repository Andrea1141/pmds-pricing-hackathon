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
    """1-unit price-ladder listing strategy with early discovery probes."""

    ladder_low: float = 0.88
    monopoly_low: float = 0.94
    crowded_low: float = 0.80
    ladder_high: float = 1.18
    monopoly_high: float = 1.26
    crowded_high: float = 1.12
    early_low: float = 0.85
    early_high: float = 1.38
    probe_mult: float = 0.74
    max_listings_per_good: int = 36

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
        """Return 1-unit ladder `Listing` offers for `goods_config.finished_goods`."""
        early = state.turn <= ctx.explore_turns
        inv = {k: int(v) for k, v in state.inventory.items()}
        is_crowded = ctx.active_team_count >= 10
        denom_rivals = max(1, ctx.active_team_count - 1)
        eg_win = 2 if state.turn > 40 else 3

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

            curr_g = max(0, inv.get(g, 0))
            exp_prod = max(0, ctx.expected_produced.get(g, ctx.need_build.get(g, 0)))
            planned = curr_g + exp_prod
            if planned <= 0:
                continue

            floor = ctx.active_floor(g)
            rival_cnt = ctx.rival_holders.get(g, 0)
            crowding = rival_cnt / float(denom_rivals)

            if early and ctx.total_sold.get(g, 0) == 0:
                low, high = self.early_low, self.early_high
            elif ctx.rival_stock.get(g, 0) == 0:
                low, high = self.monopoly_low, self.monopoly_high
            elif is_crowded and crowding >= 0.35 and curr_g >= 2:
                low, high = self.crowded_low, self.crowded_high
            else:
                low, high = self.ladder_low, self.ladder_high

            # Endgame clearance multiplier so final-turn stock converts to cash.
            eg_mult = 1.0
            if ctx.remaining_turns <= eg_win and curr_g >= 1:
                if ctx.remaining_turns == 1:
                    eg_mult = 0.65
                elif ctx.remaining_turns == 2:
                    eg_mult = 0.78
                else:
                    eg_mult = 0.88

            n = planned
            prices: list[float] = []
            for i in range(n):
                frac = 0.5 if n == 1 else i / float(n - 1)
                p = ref * eg_mult * (low + (high - low) * frac)
                prices.append(self._round_price(max(p, floor, 0.01)))

            # Early discovery probe: if a good has never sold by turn >= 2, replace the
            # lowest rung with a bounded probe so low-WTP markets are discovered quickly.
            if state.turn >= 2 and ctx.total_sold.get(g, 0) == 0 and n >= 3:
                probe_floor = max(floor, ctx.good_reserve_cost.get(g, 0.0) * 1.12)
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

            # Catchall buffer in case bonus units assemble beyond expected fill.
            top_p = ladder[-1] if ladder else ref
            catchall_p = self._round_price(max(top_p * 1.02, ref * eg_mult * 1.05, floor, 0.01))
            listings.append(Listing(g, 6, float(catchall_p)))
            my_prices.add(round(float(catchall_p), 4))

        ctx.last_ladder = last_ladder
        ctx.my_list_prices = my_prices
        return listings
