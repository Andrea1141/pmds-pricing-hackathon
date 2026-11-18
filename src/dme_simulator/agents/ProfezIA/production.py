"""Phase 2: Production / recipe assembly strategy (Owner: Persona 3).

Keep all production logic and production-specific internal state inside this
module.
"""

from __future__ import annotations

from dme_simulator.agents import AssemblyOrder, Observation
from dme_simulator.config import GoodsConfig

from .state import MarketContext


class ProductionStrategy:
    """Computes ordered assembly instructions for raw materials -> finished goods."""

    def __init__(self) -> None:
        # Store production-specific state here if needed.
        pass

    def decide_assembly(
        self,
        state: Observation,
        goods_config: GoodsConfig,
        ctx: MarketContext,
    ) -> list[AssemblyOrder]:
        """Return ordered `AssemblyOrder` list for this turn."""
        return [AssemblyOrder(name, 1) for name in goods_config.recipes]
