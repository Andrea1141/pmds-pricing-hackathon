"""Competitive configuration-independent sparring opponents for local benchmarking.

Lives strictly under `scripts/` (outside `src/dme_simulator/agents/`) so these
bots NEVER interfere with automatic bundle discovery or competition submissions.
"""

from __future__ import annotations

from math import ceil, log1p

from dme_simulator.agents import (
    AssemblyOrder,
    BaseAgent,
    Bid,
    Decision,
    Listing,
    Observation,
)
from dme_simulator.config import GoodsConfig


class _SparringBase(BaseAgent):
    """Shared incremental tracking utilities for competitive sparring bots."""

    def __init__(self) -> None:
        self._seen_auc = 0
        self._seen_sales = 0
        self._team_id: str | None = None
        self._mat_cost: dict[str, float] = {}
        self._sale_price: dict[str, float] = {}
        self._sold_count: dict[str, int] = {}
        self._sold_ema: dict[str, float] = {}
        self._unsold_streak: dict[str, int] = {}
        self._stockout_streak: dict[str, int] = {}
        self._fill_rate: dict[str, float] = {}
        self._prev_inv: dict[str, int] | None = None
        self._prev_bids: list[Bid] = []
        self._prev_reserves: dict[str, float] = {}
        self._last_anchor_price: dict[str, float] = {}

    def _observe(self, state: Observation, goods_config: GoodsConfig) -> None:
        new_auc = state.auction_transactions[self._seen_auc :]
        self._seen_auc = len(state.auction_transactions)
        new_sales = state.sales_transactions[self._seen_sales :]
        self._seen_sales = len(state.sales_transactions)

        if self._team_id is None:
            matches = [
                tid
                for tid, inv in state.public_inventories.items()
                if inv == state.inventory
                and abs(state.public_capital.get(tid, -1.0) - state.capital) < 1e-6
            ]
            if len(matches) == 1:
                self._team_id = matches[0]
            else:
                actors = {t.buyer for t in new_auc} | {s.seller for s in new_sales}
                narrow = (set(matches) if matches else set(state.public_inventories)) & actors
                if len(narrow) == 1:
                    self._team_id = next(iter(narrow))

        cur_reserves: dict[str, float] = {}
        for lot in state.auction_lots:
            cur_reserves[lot.good] = lot.reserve_price
            self._mat_cost.setdefault(lot.good, lot.reserve_price)
            self._fill_rate.setdefault(lot.good, 0.6)

        for tx in new_auc:
            prev = self._mat_cost.get(tx.good, tx.price)
            self._mat_cost[tx.good] = 0.65 * prev + 0.35 * tx.price

        for s in new_sales:
            prev = self._sale_price.get(s.good, 0.0)
            self._sale_price[s.good] = s.price if prev <= 0.0 else 0.65 * prev + 0.35 * s.price

        own_won: dict[str, int] = {}
        own_sold: dict[str, int] = {}
        for tx in new_auc:
            if self._team_id is None or tx.buyer == self._team_id:
                own_won[tx.good] = own_won.get(tx.good, 0) + tx.quantity
        for s in new_sales:
            if self._team_id is None or s.seller == self._team_id:
                own_sold[s.good] = own_sold.get(s.good, 0) + s.quantity
                self._sold_count[s.good] = self._sold_count.get(s.good, 0) + s.quantity

        if self._prev_inv is not None:
            for g in goods_config.finished_goods:
                prev_q = self._prev_inv.get(g, 0)
                curr_q = state.inventory.get(g, 0)
                sold_q = own_sold.get(g, 0)
                prod_q = max(0, curr_q - prev_q + sold_q)
                avail = prev_q + prod_q
                if avail > 0:
                    prev_ema = self._sold_ema.get(g, 0.0)
                    self._sold_ema[g] = float(sold_q) if prev_ema <= 0.0 else 0.65 * prev_ema + 0.35 * sold_q
                    if sold_q >= avail or (sold_q > 0 and curr_q == 0):
                        self._stockout_streak[g] = self._stockout_streak.get(g, 0) + 1
                        self._unsold_streak[g] = 0
                    elif sold_q == 0 and curr_q > 0:
                        self._unsold_streak[g] = self._unsold_streak.get(g, 0) + 1
                        self._stockout_streak[g] = 0
                    else:
                        self._stockout_streak[g] = 0
                        self._unsold_streak[g] = 0

            req_by_mat: dict[str, int] = {}
            for b in self._prev_bids:
                if b.quantity > 0 and b.price_per_unit >= self._prev_reserves.get(b.good, 0.0) - 1e-9:
                    req_by_mat[b.good] = req_by_mat.get(b.good, 0) + b.quantity
            for m, req in req_by_mat.items():
                obs_f = min(1.0, own_won.get(m, 0) / req)
                self._fill_rate[m] = 0.65 * self._fill_rate.get(m, 0.6) + 0.35 * obs_f

        self._prev_inv = dict(state.inventory)
        self._prev_reserves = cur_reserves

    def _recipe_cost(self, recipe_name: str, goods_config: GoodsConfig) -> float:
        rec = goods_config.recipes[recipe_name]
        return sum(self._mat_cost.get(m, 0.0) * u for m, u in rec.inputs.items())

    def _good_cost(self, good: str, goods_config: GoodsConfig) -> float:
        costs = [
            self._recipe_cost(rn, goods_config)
            for rn, rec in goods_config.recipes.items()
            if rec.output_good == good
        ]
        return min(costs) if costs else 0.0

    def _est_price(self, good: str, goods_config: GoodsConfig, default_mult: float = 4.5) -> float:
        if self._sale_price.get(good, 0.0) > 0.0:
            return self._sale_price[good]
        c = self._good_cost(good, goods_config)
        if c > 0.0:
            all_costs = [self._good_cost(g, goods_config) for g in goods_config.finished_goods]
            pos = [x for x in all_costs if x > 0.0]
            avg_c = sum(pos) / len(pos) if pos else c
            return c * default_mult * (1.0 + 0.35 * log1p(c / max(1e-6, avg_c)))
        return 100.0

    def _competitor_finished_stock(self, state: Observation, good: str) -> int:
        total = 0
        for tid, inv in state.public_inventories.items():
            if tid == self._team_id or state.public_capital.get(tid, 0.0) < 0.0:
                continue
            total += inv.get(good, 0)
        return total

    def _best_recipes_by_margin(self, goods_config: GoodsConfig) -> list[tuple[str, float, float]]:
        ranked: list[tuple[str, float, float]] = []
        for r_name, rec in goods_config.recipes.items():
            cost = self._recipe_cost(r_name, goods_config)
            rev = self._est_price(rec.output_good, goods_config)
            margin = rev - cost
            denom = max(1.0, cost, float(sum(rec.inputs.values())))
            ranked.append((r_name, margin, margin / denom))
        ranked.sort(key=lambda item: (-item[2], -item[1]))
        return ranked

    def _greedy_same_turn_assembly(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        bids: list[Bid],
        ranked_recipes: list[tuple[str, float, float]],
        cap_per_good: int,
    ) -> list[AssemblyOrder]:
        lot_map = {lot.good: lot for lot in state.auction_lots}
        pool = dict(state.inventory)
        for b in bids:
            lot = lot_map.get(b.good)
            if lot is not None and b.quantity > 0 and b.price_per_unit >= lot.reserve_price - 1e-9:
                pool[b.good] = pool.get(b.good, 0) + min(lot.quantity, b.quantity)

        shelf = {g: state.inventory.get(g, 0) for g in goods_config.finished_goods}
        orders: list[AssemblyOrder] = []
        for r_name, margin, _ in ranked_recipes:
            if margin <= 0.0:
                continue
            rec = goods_config.recipes[r_name]
            good = rec.output_good
            room = max(0, cap_per_good - shelf.get(good, 0))
            if room <= 0:
                continue
            can_make = min(pool.get(m, 0) // u for m, u in rec.inputs.items())
            qty = min(can_make, room)
            if qty > 0:
                orders.append(AssemblyOrder(r_name, qty))
                for m, u in rec.inputs.items():
                    pool[m] -= u * qty
                shelf[good] = shelf.get(good, 0) + qty
        return orders


class SmartMarginBot(_SparringBase):
    """Disciplined ROI bidder, same-turn margin assembler, adaptive pricer."""

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        self._observe(state, goods_config)
        ranked = self._best_recipes_by_margin(goods_config)
        lot_map = {lot.good: lot for lot in state.auction_lots}

        # Target units scaled by market lot volume & active teams.
        n_teams = max(1, len(state.public_inventories))
        total_lots = sum(lot.quantity for lot in state.auction_lots)
        avg_in = sum(sum(r.inputs.values()) for r in goods_config.recipes.values()) / max(1, len(goods_config.recipes))
        target_per_recipe = max(2, min(14, ceil(total_lots / max(1.0, avg_in * n_teams * 1.5))))

        wanted: dict[str, int] = {m: 0 for m in goods_config.raw_materials}
        max_wtp: dict[str, float] = {m: 0.0 for m in goods_config.raw_materials}
        top_k = max(1, ceil(len(ranked) * 0.65))
        for r_name, margin, _ in ranked[:top_k]:
            if margin <= 0.0:
                continue
            rec = goods_config.recipes[r_name]
            good = rec.output_good
            unsold = self._unsold_streak.get(good, 0)
            vel = self._sold_ema.get(good, 0.0)
            desired_g = max(1, ceil(vel * 1.5 + 1.0)) if self._sold_count.get(good, 0) > 0 else target_per_recipe
            if unsold >= 2:
                desired_g = 1
            need_g = max(0, desired_g - state.inventory.get(good, 0))
            if need_g <= 0:
                continue
            tot_in = max(1, sum(rec.inputs.values()))
            rev = self._est_price(good, goods_config)
            for m, u in rec.inputs.items():
                wanted[m] = wanted.get(m, 0) + u * need_g
                max_wtp[m] = max(max_wtp.get(m, 0.0), (rev * 0.65) / tot_in)

        budget = state.capital * 0.62
        bid_prices: dict[str, float] = {}
        bid_qtys: dict[str, int] = {}
        planned_spend = 0.0
        for lot in state.auction_lots:
            if lot.quantity <= 0:
                continue
            need = max(0, wanted.get(lot.good, 0) - state.inventory.get(lot.good, 0))
            if need <= 0:
                continue
            fill = self._fill_rate.get(lot.good, 0.6)
            mult = 1.38 if fill < 0.45 else (1.14 if fill > 0.85 else 1.24)
            base_p = max(lot.reserve_price, self._mat_cost.get(lot.good, lot.reserve_price))
            cap_p = max(lot.reserve_price * 1.02, max_wtp.get(lot.good, base_p * 1.3))
            price = min(cap_p, max(lot.reserve_price * 1.02, base_p * mult))
            if lot.reserve_price <= 0.0:
                price = max(0.01, min(cap_p, 1.0))
            qty = min(lot.quantity, need)
            bid_prices[lot.good] = price
            bid_qtys[lot.good] = qty
            planned_spend += price * qty

        scale = min(1.0, budget / planned_spend) if planned_spend > budget and planned_spend > 0 else 1.0
        bids = [
            Bid(m, max(1, int(q * scale)), bid_prices[m])
            for m, q in bid_qtys.items()
            if int(q * scale) > 0
        ]
        self._prev_bids = bids

        assembly = self._greedy_same_turn_assembly(
            state, goods_config, bids, ranked, cap_per_good=max(4, target_per_recipe * 2)
        )

        # Adaptive pricing based on sellout vs unsold streak.
        listings: list[Listing] = []
        for g in goods_config.finished_goods:
            cost = self._good_cost(g, goods_config)
            anchor = self._last_anchor_price.get(g, self._est_price(g, goods_config, default_mult=5.0))
            if self._stockout_streak.get(g, 0) >= 1:
                anchor *= 1.10
            elif self._unsold_streak.get(g, 0) >= 1:
                anchor *= 0.86 if self._unsold_streak[g] >= 2 else 0.92
            elif self._sale_price.get(g, 0.0) > 0.0:
                anchor = 0.7 * anchor + 0.3 * self._sale_price[g] * 1.04
            floor = max(1.0, cost * 1.15)
            price = max(floor, anchor)
            self._last_anchor_price[g] = price
            listings.append(Listing(g, 10_000, price))

        return Decision(bids=bids, assembly=assembly, listings=listings)


class BottleneckSniperBot(_SparringBase):
    """Corners the scarcest raw materials and charges monopoly premiums on stockouts."""

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        self._observe(state, goods_config)
        ranked = self._best_recipes_by_margin(goods_config)
        if not state.auction_lots:
            return Decision()

        # Rank raw materials by scarcity (highest recipe usage / lot quantity).
        usage: dict[str, int] = {m: 0 for m in goods_config.raw_materials}
        for rec in goods_config.recipes.values():
            for m, u in rec.inputs.items():
                usage[m] = usage.get(m, 0) + u
        scarcity_score = {
            lot.good: usage.get(lot.good, 1) / max(1.0, float(lot.quantity))
            for lot in state.auction_lots
        }
        top_scarce = set(
            sorted(scarcity_score, key=lambda m: -scarcity_score[m])[: max(1, len(scarcity_score) // 2)]
        )

        n_teams = max(1, len(state.public_inventories))
        budget = state.capital * 0.58
        # Economic upper bound per material based on best recipe revenue.
        max_wtp: dict[str, float] = {}
        for r_name, margin, _ in ranked:
            if margin <= 0.0:
                continue
            rec = goods_config.recipes[r_name]
            rev = self._est_price(rec.output_good, goods_config)
            tot_in = max(1, sum(rec.inputs.values()))
            for m in rec.inputs:
                max_wtp[m] = max(max_wtp.get(m, 0.0), (rev * 0.55) / tot_in)

        total_unsold = sum(state.inventory.get(g, 0) for g in goods_config.finished_goods)
        throttle = 0.5 if total_unsold > 12 else 1.0

        bids_raw: list[tuple[str, int, float]] = []
        spend = 0.0
        for lot in state.auction_lots:
            if lot.quantity <= 0:
                continue
            is_scarce = lot.good in top_scarce
            share = 0.35 if is_scarce else 0.18
            target_q = max(1, ceil(lot.quantity * share * throttle / max(1.0, n_teams * 0.45)))
            need = max(0, target_q - state.inventory.get(lot.good, 0))
            if need <= 0:
                continue
            mult = 1.38 if is_scarce else 1.15
            base_p = max(lot.reserve_price, self._mat_cost.get(lot.good, lot.reserve_price))
            cap_p = max_wtp.get(lot.good, base_p * 1.4)
            if cap_p < lot.reserve_price:
                continue
            price = min(cap_p, max(lot.reserve_price * 1.03, base_p * mult))
            if lot.reserve_price <= 0.0:
                price = max(0.05, min(cap_p, 1.0 if is_scarce else 0.4))
            qty = min(lot.quantity, need)
            bids_raw.append((lot.good, qty, price))
            spend += qty * price

        scale = min(1.0, budget / spend) if spend > budget and spend > 0 else 1.0
        bids = [
            Bid(m, max(1, int(q * scale)), p)
            for m, q, p in bids_raw
            if int(q * scale) > 0
        ]
        self._prev_bids = bids

        assembly = self._greedy_same_turn_assembly(
            state, goods_config, bids, ranked, cap_per_good=12
        )

        listings: list[Listing] = []
        for g in goods_config.finished_goods:
            cost = self._good_cost(g, goods_config)
            anchor = self._last_anchor_price.get(g, self._est_price(g, goods_config, default_mult=5.5))
            if self._stockout_streak.get(g, 0) >= 1:
                anchor *= 1.12
            elif self._unsold_streak.get(g, 0) >= 1:
                anchor *= 0.85
            # Monopoly surge when rivals have zero finished stock of this good.
            rival_stock = self._competitor_finished_stock(state, g)
            price = anchor * 1.22 if rival_stock == 0 and self._unsold_streak.get(g, 0) == 0 else anchor
            price = max(max(1.0, cost * 1.20), price)
            self._last_anchor_price[g] = anchor
            listings.append(Listing(g, 10_000, price))

        return Decision(bids=bids, assembly=assembly, listings=listings)


class MultiTierPricerBot(_SparringBase):
    """Fill-rate-adaptive bidder + 3-tier price-discriminating seller."""

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        self._observe(state, goods_config)
        ranked = self._best_recipes_by_margin(goods_config)
        n_teams = max(1, len(state.public_inventories))
        budget = state.capital * 0.65

        wanted: dict[str, int] = {m: 0 for m in goods_config.raw_materials}
        for r_name, margin, _ in ranked:
            if margin <= 0.0:
                continue
            rec = goods_config.recipes[r_name]
            for m, u in rec.inputs.items():
                wanted[m] = wanted.get(m, 0) + u * 2

        bids_raw: list[tuple[str, int, float]] = []
        spend = 0.0
        for lot in state.auction_lots:
            if lot.quantity <= 0:
                continue
            fair_q = max(wanted.get(lot.good, 1), ceil(lot.quantity / max(1, n_teams)))
            need = max(0, fair_q - state.inventory.get(lot.good, 0))
            if need <= 0:
                continue
            fill = self._fill_rate.get(lot.good, 0.6)
            mult = 1.42 if fill < 0.40 else (1.10 if fill > 0.85 else 1.25)
            base_p = max(lot.reserve_price, self._mat_cost.get(lot.good, lot.reserve_price))
            price = max(lot.reserve_price * 1.03, base_p * mult)
            if lot.reserve_price <= 0.0:
                price = 0.25
            qty = min(lot.quantity, need)
            bids_raw.append((lot.good, qty, price))
            spend += qty * price

        scale = min(1.0, budget / spend) if spend > budget and spend > 0 else 1.0
        bids = [
            Bid(m, max(1, int(q * scale)), p)
            for m, q, p in bids_raw
            if int(q * scale) > 0
        ]
        self._prev_bids = bids

        assembly = self._greedy_same_turn_assembly(
            state, goods_config, bids, ranked, cap_per_good=14
        )
        planned_prod: dict[str, int] = {}
        for o in assembly:
            g = goods_config.recipes[o.recipe_name].output_good
            planned_prod[g] = planned_prod.get(g, 0) + o.quantity

        # Split stock across 3 price tiers (premium / mid / clearance) per good.
        listings: list[Listing] = []
        for g in goods_config.finished_goods:
            cost = self._good_cost(g, goods_config)
            anchor = self._last_anchor_price.get(g, self._est_price(g, goods_config, default_mult=5.0))
            if self._stockout_streak.get(g, 0) >= 1:
                anchor *= 1.10
            elif self._unsold_streak.get(g, 0) >= 1:
                anchor *= 0.86
            floor = max(1.0, cost * 1.12)
            anchor = max(floor, anchor)
            self._last_anchor_price[g] = anchor

            est_stock = state.inventory.get(g, 0) + planned_prod.get(g, 0)
            if est_stock <= 1:
                listings.append(Listing(g, 10_000, anchor))
            else:
                q_prem = max(1, int(est_stock * 0.35))
                q_mid = max(1, int(est_stock * 0.40))
                listings.append(Listing(g, q_prem, max(floor, anchor * 1.24)))
                listings.append(Listing(g, q_mid, anchor))
                listings.append(Listing(g, 10_000, max(floor, anchor * 0.82)))

        return Decision(bids=bids, assembly=assembly, listings=listings)


class HighTurnoverBot(_SparringBase):
    """High-volume, tight-margin competitor targeting fast cash velocity."""

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        self._observe(state, goods_config)
        ranked = self._best_recipes_by_margin(goods_config)
        n_teams = max(1, len(state.public_inventories))
        budget = state.capital * 0.55

        max_wtp: dict[str, float] = {}
        wanted: dict[str, int] = {m: 0 for m in goods_config.raw_materials}
        for r_name, margin, _ in ranked:
            if margin <= 0.0:
                continue
            rec = goods_config.recipes[r_name]
            good = rec.output_good
            rev = self._est_price(good, goods_config)
            tot_in = max(1, sum(rec.inputs.values()))
            vel = self._sold_ema.get(good, 0.0)
            desired_g = max(2, ceil(vel * 1.4 + 1.0)) if self._sold_count.get(good, 0) > 0 else 2
            if self._unsold_streak.get(good, 0) >= 2:
                desired_g = 0
            need_g = max(0, desired_g - state.inventory.get(good, 0))
            for m, u in rec.inputs.items():
                wanted[m] = wanted.get(m, 0) + u * need_g
                max_wtp[m] = max(max_wtp.get(m, 0.0), (rev * 0.50) / tot_in)

        bids_raw: list[tuple[str, int, float]] = []
        spend = 0.0
        for lot in state.auction_lots:
            if lot.quantity <= 0:
                continue
            target_q = min(max(2, ceil(lot.quantity / max(1.0, n_teams * 0.9))), max(2, wanted.get(lot.good, 2)))
            need = max(0, target_q - state.inventory.get(lot.good, 0))
            if need <= 0:
                continue
            base_p = max(lot.reserve_price, self._mat_cost.get(lot.good, lot.reserve_price))
            cap_p = max_wtp.get(lot.good, base_p * 1.25)
            if cap_p < lot.reserve_price:
                continue
            price = min(cap_p, max(lot.reserve_price * 1.03, base_p * 1.20))
            if lot.reserve_price <= 0.0:
                price = max(0.02, min(cap_p, 0.2))
            qty = min(lot.quantity, need)
            bids_raw.append((lot.good, qty, price))
            spend += qty * price

        scale = min(1.0, budget / spend) if spend > budget and spend > 0 else 1.0
        bids = [
            Bid(m, max(1, int(q * scale)), p)
            for m, q, p in bids_raw
            if int(q * scale) > 0
        ]
        self._prev_bids = bids

        assembly = self._greedy_same_turn_assembly(
            state, goods_config, bids, ranked, cap_per_good=18
        )

        listings: list[Listing] = []
        for g in goods_config.finished_goods:
            cost = self._good_cost(g, goods_config)
            anchor = self._last_anchor_price.get(g, self._est_price(g, goods_config, default_mult=3.6))
            if self._stockout_streak.get(g, 0) >= 2:
                anchor *= 1.07
            elif self._unsold_streak.get(g, 0) >= 1:
                anchor *= 0.82
            elif self._sale_price.get(g, 0.0) > 0.0:
                anchor = 0.6 * anchor + 0.4 * self._sale_price[g] * 0.94
            floor = max(1.0, cost * 1.12)
            price = max(floor, anchor)
            self._last_anchor_price[g] = price
            listings.append(Listing(g, 10_000, price))

        return Decision(bids=bids, assembly=assembly, listings=listings)


def build_sparring_roster() -> dict[str, BaseAgent]:
    """Return fresh instances of the 4 competitive sparring opponents."""
    return {
        "SmartMarginBot": SmartMarginBot(),
        "BottleneckSniperBot": BottleneckSniperBot(),
        "MultiTierPricerBot": MultiTierPricerBot(),
        "HighTurnoverBot": HighTurnoverBot(),
    }
