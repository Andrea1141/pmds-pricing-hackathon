"""Shared market state and incremental history tracking (Owner: Persona 3).

Updated once at the start of every turn by `agent.py` before calling
`bidding`, `production`, and `pricing`.
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
    """Shared read-only market signals computed from public observations."""

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

    def update(self, state: Observation, goods_config: GoodsConfig) -> None:
        """Advance shared market estimates using the latest turn observation."""
        self.turn = state.turn
        self.planned_bids = []
        self.planned_assembly = []

        if self.team_id is None:
            self.team_id = self._infer_team_id(state)

        # Slice only newly arrived transactions since last turn.
        self.new_auction_txs = state.auction_transactions[self._seen_auction_count :]
        self._seen_auction_count = len(state.auction_transactions)

        self.new_sales_txs = state.sales_transactions[self._seen_sales_count :]
        self._seen_sales_count = len(state.sales_transactions)

        # Record current auction lots and seed material costs from reserve prices.
        for lot in state.auction_lots:
            self.reserve_price_history.setdefault(lot.good, []).append(lot.reserve_price)
            self.lot_quantity_history.setdefault(lot.good, []).append(lot.quantity)
            self.avg_material_cost.setdefault(lot.good, lot.reserve_price)

        # Update running material cost estimates from new auction transactions.
        alpha = 0.35
        for tx in self.new_auction_txs:
            prev = self.avg_material_cost.get(tx.good, tx.price)
            self.avg_material_cost[tx.good] = (1.0 - alpha) * prev + alpha * tx.price

        # Update running finished-good sale price estimates from new sales.
        for sale in self.new_sales_txs:
            prev = self.est_sale_price.get(sale.good, sale.price)
            self.est_sale_price[sale.good] = (1.0 - alpha) * prev + alpha * sale.price

        # Ensure every raw material and finished good has a fallback key.
        for raw in goods_config.raw_materials:
            self.avg_material_cost.setdefault(raw, 0.0)
        for good in goods_config.finished_goods:
            self.est_sale_price.setdefault(good, 0.0)

    def estimate_recipe_cost(self, recipe_name: str, goods_config: GoodsConfig) -> float:
        """Return estimated raw-material unit cost for one unit of `recipe_name`."""
        recipe = goods_config.recipes[recipe_name]
        return sum(
            self.avg_material_cost.get(mat, 0.0) * qty
            for mat, qty in recipe.inputs.items()
        )

    @staticmethod
    def _infer_team_id(state: Observation) -> str | None:
        matches = [
            tid
            for tid, inv in state.public_inventories.items()
            if inv == state.inventory
            and abs(state.public_capital.get(tid, -1.0) - state.capital) < 1e-6
        ]
        return matches[0] if len(matches) == 1 else None
