"""38-team realistic tournament benchmark mirroring the official competition arena.

Run from the repository root with:
    uv run python scripts/eval_tournament38.py --seeds 5
    uv run python scripts/eval_tournament38.py --seeds 3 --quick
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from sparring_opponents import (  # noqa: E402
    BottleneckSniperBot,
    HighTurnoverBot,
    MultiTierPricerBot,
    SmartMarginBot,
    _SparringBase,
)

from dme_simulator.agents import (  # noqa: E402
    AssemblyOrder,
    BaseAgent,
    Bid,
    Decision,
    Listing,
    Observation,
)
from dme_simulator.agents.loader import (  # noqa: E402
    load_simulation_agents,
    resolve_agent_directories,
)
from dme_simulator.config import (  # noqa: E402
    EnvironmentConfig,
    GoodsConfig,
    SupplierAdaptationConfig,
    SupplierConfig,
    load_config,
)
from dme_simulator.runner import run_simulation  # noqa: E402


class EarlyBlitzerRival(_SparringBase):
    """Aggressive rival simulating top leaderboard blitzers (spends heavy Turns 1-6)."""

    def __init__(self, blitz_mult: float = 2.8, price_factor: float = 1.0) -> None:
        super().__init__()
        self._blitz_mult = blitz_mult
        self._price_factor = price_factor

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        self._observe(state, goods_config)
        ranked = self._best_recipes_by_margin(goods_config)
        top_recipes = [r for r in ranked[:2] if r[1] > 0.0]

        is_early = state.turn <= 6
        fin_inv = sum(state.inventory.get(g, 0) for g in goods_config.finished_goods)
        wanted: dict[str, int] = {m: 0 for m in goods_config.raw_materials}

        if is_early:
            for r_name, _, _ in top_recipes:
                for m, u in goods_config.recipes[r_name].inputs.items():
                    wanted[m] = wanted.get(m, 0) + u * 4
        elif fin_inv < 6 and state.turn <= 36:
            for r_name, margin, _ in ranked[:3]:
                if margin <= 0.0:
                    continue
                rec = goods_config.recipes[r_name]
                if state.inventory.get(rec.output_good, 0) <= 1:
                    for m, u in rec.inputs.items():
                        wanted[m] = wanted.get(m, 0) + u * 2

        budget = state.capital * (0.85 if is_early else 0.45)
        raw_bids: list[tuple[str, int, float]] = []
        spend = 0.0
        for lot in state.auction_lots:
            if lot.quantity <= 0:
                continue
            need = max(0, wanted.get(lot.good, 0) - state.inventory.get(lot.good, 0))
            if need <= 0:
                continue
            mult = self._blitz_mult if is_early else 1.35
            price = max(0.2, lot.reserve_price * mult)
            qty = min(lot.quantity, need)
            raw_bids.append((lot.good, qty, price))
            spend += qty * price

        scale = min(1.0, budget / spend) if spend > budget and spend > 0 else 1.0
        bids = [
            Bid(m, max(1, int(q * scale)), p)
            for m, q, p in raw_bids
            if int(q * scale) > 0
        ]
        self._prev_bids = bids

        assembly = self._greedy_same_turn_assembly(
            state, goods_config, bids, ranked, cap_per_good=8
        )

        listings: list[Listing] = []
        for g in goods_config.finished_goods:
            cost = self._good_cost(g, goods_config)
            anchor = self._last_anchor_price.get(
                g, self._est_price(g, goods_config, default_mult=5.0) * self._price_factor
            )
            if self._stockout_streak.get(g, 0) >= 1:
                anchor *= 1.08
            elif self._unsold_streak.get(g, 0) >= 1:
                anchor *= 0.85
            floor = max(1.0, cost * 1.15)
            anchor = max(floor, anchor)
            self._last_anchor_price[g] = anchor
            # Split into 5 fine 1-unit steps + catchall
            for mult in (0.94, 0.99, 1.04, 1.10, 1.18):
                listings.append(Listing(g, 1, max(floor, anchor * mult)))
            listings.append(Listing(g, 1000, max(floor, anchor * 0.90)))

        return Decision(bids=bids, assembly=assembly, listings=listings)


def _build_38_team_roster(
    config: EnvironmentConfig, config_path: Path, target_agent: str
) -> dict[str, BaseAgent]:
    directories = resolve_agent_directories(config_path, config.agents.directory, None)
    discovered = load_simulation_agents(directories, config.agents.include)
    if target_agent not in discovered:
        raise SystemExit(f"Target agent {target_agent!r} not found in {list(discovered)}")

    # 38 total teams: ProfezIA + 3 built-ins + 34 diverse competitive opponents
    agents: dict[str, BaseAgent] = dict(discovered)
    for i in range(8):
        agents[f"EarlyBlitzer_{i}"] = EarlyBlitzerRival(
            blitz_mult=2.2 + 0.25 * (i % 5),
            price_factor=0.88 + 0.06 * (i % 5),
        )
    for i in range(9):
        agents[f"MultiTier_{i}"] = MultiTierPricerBot()
        agents[f"SmartMargin_{i}"] = SmartMarginBot()
    for i in range(4):
        agents[f"HighTurn_{i}"] = HighTurnoverBot()
        agents[f"Sniper_{i}"] = BottleneckSniperBot()
    return agents


def _compute_ranks(capitals: dict[str, float]) -> dict[str, float]:
    ordered = sorted(capitals.items(), key=lambda item: -item[1])
    ranks: dict[str, float] = {}
    idx = 0
    while idx < len(ordered):
        end = idx + 1
        while end < len(ordered) and abs(ordered[end][1] - ordered[idx][1]) < 1e-6:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for pos in range(idx, end):
            ranks[ordered[pos][0]] = avg_rank
        idx = end
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description="38-team crowded tournament benchmark")
    parser.add_argument("--seeds", type=int, default=3, help="seeds per scenario (default: 3)")
    parser.add_argument("--agent", type=str, default="ProfezIA", help="target agent ID")
    parser.add_argument("--quick", action="store_true", help="only run default 38-team room")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    default_path = repo_root / "config" / "default.yaml"
    scaled_path = repo_root / "config" / "scaled_market.yaml"
    default_cfg = load_config(default_path)
    scaled_cfg = load_config(scaled_path)

    tight_sup = dict(default_cfg.suppliers)
    tight_sup["DISPLAY"] = SupplierConfig(
        production_min=8,
        production_max=14,
        base_price=35.0,
        stock_capacity=40,
        price_stock_factor=0.25,
        price_floor_multiplier=0.55,
        adaptation=SupplierAdaptationConfig(enabled=False),
    )
    bottleneck_cfg = replace(default_cfg, turns=35, seed=91, suppliers=tight_sup)
    bottleneck_cfg.validate()
    long60_cfg = replace(default_cfg, turns=60, seed=92)

    scenarios = [("default_38", default_cfg, default_path)]
    if not args.quick:
        scenarios.extend(
            [
                ("scaled_38", scaled_cfg, scaled_path),
                ("bottleneck_38", bottleneck_cfg, default_path),
                ("long60_38", long60_cfg, default_path),
            ]
        )

    print(f"38-Team Tournament Benchmark for {args.agent} ({args.seeds} seed(s)/scenario)")
    header = (
        f"{'Scenario':<15} {'Rank/38':>8} {'Capital':>11} {'BestRival':>11} "
        f"{'EarlyCF(1-8)':>13} {'Unsold':>7} {'Raw':>6} {'Max ms':>8} {'Bankr':>5}"
    )
    print(header)
    print("-" * len(header))

    all_ranks: list[float] = []
    worst_ms = 0.0
    for name, cfg, cfg_path in scenarios:
        quiet_cfg = replace(
            cfg,
            output=replace(
                cfg.output,
                render_plots=False,
                progress_interval_turns=0,
                dashboard="none",
            ),
        )
        ranks, caps, best_rivals, early_cfs, unsolds, raws, max_ms_list = [], [], [], [], [], [], []
        bankruptcies = 0
        for s in range(args.seeds):
            agents = _build_38_team_roster(quiet_cfg, cfg_path, args.agent)
            sim = run_simulation(
                quiet_cfg, agents, seed_override=quiet_cfg.seed + s, profile=True, quiet=True
            )
            final_caps = {tid: st.capital for tid, st in sim.teams_state.items()}
            r_map = _compute_ranks(final_caps)
            ranks.append(r_map[args.agent])
            st = sim.teams_state[args.agent]
            caps.append(st.capital)
            if st.bankrupt:
                bankruptcies += 1
            best_rivals.append(max(c for t, c in final_caps.items() if t != args.agent))
            rev_early = sum(sim.history.revenue[args.agent][:8])
            spend_early = sum(sim.history.auction_spend[args.agent][:8])
            early_cfs.append(rev_early - spend_early)
            unsolds.append(sum(st.inventory.get(g, 0) for g in quiet_cfg.goods.finished_goods))
            raws.append(sum(st.inventory.get(m, 0) for m in quiet_cfg.goods.raw_materials))
            durs = sim.history.diagnostics.decision_duration_seconds.get(args.agent, [])
            max_ms_list.append(max(durs) * 1000.0 if durs else 0.0)

        all_ranks.append(sum(ranks) / len(ranks))
        worst_ms = max(worst_ms, max(max_ms_list))
        print(
            f"{name:<15} {sum(ranks)/len(ranks):8.2f} {sum(caps)/len(caps):11.2f} "
            f"{sum(best_rivals)/len(best_rivals):11.2f} "
            f"{sum(early_cfs)/len(early_cfs):+13.2f} "
            f"{sum(unsolds)/len(unsolds):7.1f} {sum(raws)/len(raws):6.1f} "
            f"{max(max_ms_list):8.2f} {bankruptcies:5d}"
        )

    print("-" * len(header))
    print(
        f"Overall 38-team avg rank: {sum(all_ranks)/len(all_ranks):.2f}/38 | "
        f"Worst latency: {worst_ms:.2f} ms (limit: 100.00 ms)"
    )


if __name__ == "__main__":
    main()
