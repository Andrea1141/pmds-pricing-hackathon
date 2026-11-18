"""Multi-scenario evaluation, sparring arena, and private-stress harness for ProfezIA.

Run from the repository root with:
    uv run python scripts/eval_profezia.py
    uv run python scripts/eval_profezia.py --seeds 5
    uv run python scripts/eval_profezia.py --seeds 3 --sparring --leaderboard
    uv run python scripts/eval_profezia.py --seeds 3 --sparring --private-stress
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# Allow `from sparring_opponents import ...` regardless of invocation cwd.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from sparring_opponents import build_sparring_roster  # noqa: E402

from dme_simulator.agents import BaseAgent  # noqa: E402
from dme_simulator.agents.loader import (  # noqa: E402
    load_simulation_agents,
    resolve_agent_directories,
)
from dme_simulator.config import (  # noqa: E402
    ConsumerMarketConfig,
    EnvironmentConfig,
    GoodsConfig,
    InitialStateConfig,
    ProductionConfig,
    RecipeConfig,
    SegmentConfig,
    SupplierAdaptationConfig,
    SupplierConfig,
    load_config,
)
from dme_simulator.runner import run_simulation  # noqa: E402


def _build_standard_scenarios(
    repo_root: Path, quick: bool
) -> list[tuple[str, EnvironmentConfig, Path, str]]:
    """Return `(name, config, config_path, roster_mode)` tuples."""
    default_path = repo_root / "config" / "default.yaml"
    scaled_path = repo_root / "config" / "scaled_market.yaml"
    tut_auc_path = repo_root / "config" / "tutorial_auction.yaml"
    tut_prc_path = repo_root / "config" / "tutorial_pricing.yaml"

    default_cfg = load_config(default_path)
    scaled_cfg = load_config(scaled_path)
    tut_auc_cfg = load_config(tut_auc_path)
    tut_prc_cfg = load_config(tut_prc_path)

    cap4_cfg = replace(
        default_cfg,
        production=replace(default_cfg.production, capacity_per_turn=4),
    )
    obs_all_cfg = replace(
        default_cfg,
        agents=replace(default_cfg.agents, observe_all_transactions=True),
    )
    short12_cfg = replace(default_cfg, turns=12)

    scenarios = [
        ("default", default_cfg, default_path, "default"),
        ("scaled_market", scaled_cfg, scaled_path, "default"),
        ("stress_cap4", cap4_cfg, default_path, "default"),
        ("stress_obs_all", obs_all_cfg, default_path, "default"),
        ("stress_short12", short12_cfg, default_path, "default"),
        ("tutorial_auction", tut_auc_cfg, tut_auc_path, "default"),
        ("tutorial_pricing", tut_prc_cfg, tut_prc_path, "default"),
    ]
    if quick:
        return [
            (name, replace(cfg, turns=min(cfg.turns, 15)), path, mode)
            for name, cfg, path, mode in scenarios[:4]
        ]
    return scenarios


def _build_private_stress_scenarios(
    repo_root: Path,
) -> list[tuple[str, EnvironmentConfig, Path, str]]:
    """Synthetic unseen market geometries mirroring hard private competition scenarios."""
    default_path = repo_root / "config" / "default.yaml"
    base = load_config(default_path)

    # 1. Multiple recipes producing the SAME finished good (recipe selection test).
    multi_recipe_goods = GoodsConfig(
        raw_materials=["ALLOY", "CORE", "CELL", "GLASS"],
        finished_goods=["DRONE", "VISOR", "ROVER"],
        recipes={
            "DRONE_STD": RecipeConfig("DRONE", {"ALLOY": 1, "CORE": 1, "CELL": 1}),
            "DRONE_ALT": RecipeConfig("DRONE", {"ALLOY": 2, "CELL": 2}),
            "VISOR_STD": RecipeConfig("VISOR", {"CORE": 2, "GLASS": 2}),
            "VISOR_LITE": RecipeConfig("VISOR", {"CORE": 1, "GLASS": 3, "CELL": 1}),
            "ROVER_HEAVY": RecipeConfig("ROVER", {"ALLOY": 2, "CORE": 2, "CELL": 2, "GLASS": 1}),
        },
    )
    multi_recipe_cfg = replace(
        base,
        turns=30,
        seed=71,
        goods=multi_recipe_goods,
        suppliers={
            "ALLOY": SupplierConfig(35, 50, base_price=12.0, stock_capacity=220),
            "CORE": SupplierConfig(18, 28, base_price=30.0, stock_capacity=140),
            "CELL": SupplierConfig(30, 45, base_price=14.0, stock_capacity=200),
            "GLASS": SupplierConfig(25, 38, base_price=16.0, stock_capacity=180),
        },
        consumer_market=ConsumerMarketConfig(
            base_customers=36,
            amplitude=10,
            period_divisor=4.5,
            price_sensitivity_beta=0.015,
            segments={
                "MASS": SegmentConfig(
                    share=0.60,
                    preferences={"DRONE": 3, "VISOR": 2},
                    willingness_to_pay={"DRONE": (120.0, 260.0), "VISOR": (220.0, 420.0)},
                ),
                "PRO": SegmentConfig(
                    share=0.40,
                    preferences={"ROVER": 3, "VISOR": 2, "DRONE": 1},
                    willingness_to_pay={
                        "ROVER": (550.0, 1100.0),
                        "VISOR": (380.0, 680.0),
                        "DRONE": (200.0, 350.0),
                    },
                ),
            },
        ),
    )
    multi_recipe_cfg.validate()

    # 2. Extreme single-material bottleneck without supplier adaptation.
    tight_suppliers = dict(base.suppliers)
    tight_suppliers["DISPLAY"] = SupplierConfig(
        production_min=6,
        production_max=10,
        base_price=40.0,
        stock_capacity=30,
        price_stock_factor=0.25,
        price_floor_multiplier=0.60,
        adaptation=SupplierAdaptationConfig(enabled=False),
    )
    bottleneck_cfg = replace(base, turns=30, seed=72, suppliers=tight_suppliers)
    bottleneck_cfg.validate()

    # 3. Fierce Bertrand price war (high price_sensitivity_beta).
    bertrand_cfg = replace(
        base,
        turns=30,
        seed=73,
        consumer_market=replace(base.consumer_market, price_sensitivity_beta=0.08),
    )
    bertrand_cfg.validate()

    # 4. Severe cyclical demand crashes (amplitude almost equals base_customers).
    wave_crash_cfg = replace(
        base,
        turns=32,
        seed=74,
        consumer_market=replace(
            base.consumer_market,
            base_customers=30,
            amplitude=28,
            period_divisor=2.5,
        ),
    )
    wave_crash_cfg.validate()

    # 5. Zero reserve prices + occasional empty lots (division-by-zero / edge-case test).
    zero_suppliers = {
        mat: SupplierConfig(
            production_min=0,
            production_max=20,
            base_price=0.0,
            stock_capacity=80,
            price_stock_factor=0.0,
            price_floor_multiplier=1.0,
            adaptation=SupplierAdaptationConfig(enabled=False),
        )
        for mat in base.goods.raw_materials
    }
    zero_reserve_cfg = replace(
        base,
        turns=25,
        seed=75,
        initial_state=InitialStateConfig(starting_capital=8_000.0, initial_inventory={"CHIP": 2}),
        production=ProductionConfig(capacity_per_turn=6),
        suppliers=zero_suppliers,
    )
    zero_reserve_cfg.validate()

    # 6. Crowded 8-team tournament room (built-ins + sparring bots + ProfezIA).
    crowded_cfg = replace(
        base,
        turns=30,
        seed=76,
        consumer_market=replace(base.consumer_market, base_customers=48, amplitude=12),
    )
    crowded_cfg.validate()

    return [
        ("priv_multi_recipe", multi_recipe_cfg, default_path, "default"),
        ("priv_bottleneck", bottleneck_cfg, default_path, "default"),
        ("priv_bertrand", bertrand_cfg, default_path, "default"),
        ("priv_wave_crash", wave_crash_cfg, default_path, "default"),
        ("priv_zero_reserve", zero_reserve_cfg, default_path, "default"),
        ("priv_crowded_8", crowded_cfg, default_path, "crowded"),
    ]


def _build_roster(
    config: EnvironmentConfig,
    config_path: Path,
    target_agent: str,
    use_sparring: bool,
    roster_mode: str,
) -> dict[str, BaseAgent]:
    directories = resolve_agent_directories(config_path, config.agents.directory, None)
    discovered = load_simulation_agents(directories, config.agents.include)
    if target_agent not in discovered:
        raise SystemExit(f"Target agent {target_agent!r} not found in {list(discovered)}")

    if roster_mode == "crowded":
        roster = dict(discovered)
        roster.update(build_sparring_roster())
        return roster

    if use_sparring:
        roster = build_sparring_roster()
        roster[target_agent] = discovered[target_agent]
        return roster

    return discovered


def _compute_ranks(capitals: dict[str, float]) -> dict[str, float]:
    """Compute 1-based competition ranks (higher capital = better, ties averaged)."""
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


def _evaluate_scenario(
    config: EnvironmentConfig,
    config_path: Path,
    seeds: int,
    target_agent: str,
    use_sparring: bool,
    roster_mode: str,
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, list[float]]]:
    quiet_cfg = replace(
        config,
        output=replace(
            config.output,
            render_plots=False,
            progress_interval_turns=0,
            dashboard="none",
        ),
    )

    ranks: list[float] = []
    capitals: list[float] = []
    unsold_goods: list[int] = []
    raw_leftover: list[int] = []
    decision_times_ms: list[float] = []
    bankruptcies = 0
    best_rival_caps: list[float] = []

    team_ranks: dict[str, list[float]] = {}
    team_caps: dict[str, list[float]] = {}

    for offset in range(seeds):
        seed = quiet_cfg.seed + offset
        agents = _build_roster(
            quiet_cfg, config_path, target_agent, use_sparring, roster_mode
        )
        sim = run_simulation(
            quiet_cfg,
            agents,
            seed_override=seed,
            profile=True,
            quiet=True,
        )

        final_caps = {tid: st.capital for tid, st in sim.teams_state.items()}
        seed_ranks = _compute_ranks(final_caps)
        ranks.append(seed_ranks[target_agent])

        for tid, r_val in seed_ranks.items():
            team_ranks.setdefault(tid, []).append(r_val)
            team_caps.setdefault(tid, []).append(final_caps[tid])

        st = sim.teams_state[target_agent]
        capitals.append(st.capital)
        if st.bankrupt:
            bankruptcies += 1
        unsold_goods.append(
            sum(st.inventory.get(g, 0) for g in quiet_cfg.goods.finished_goods)
        )
        raw_leftover.append(
            sum(st.inventory.get(m, 0) for m in quiet_cfg.goods.raw_materials)
        )

        rivals = [cap for tid, cap in final_caps.items() if tid != target_agent]
        best_rival_caps.append(max(rivals) if rivals else 0.0)

        durations = sim.history.diagnostics.decision_duration_seconds.get(
            target_agent, []
        )
        decision_times_ms.extend(d * 1_000.0 for d in durations)

    summary = {
        "avg_rank": sum(ranks) / len(ranks),
        "avg_capital": sum(capitals) / len(capitals),
        "avg_best_rival": sum(best_rival_caps) / len(best_rival_caps),
        "avg_unsold": sum(unsold_goods) / len(unsold_goods),
        "avg_raw": sum(raw_leftover) / len(raw_leftover),
        "bankruptcies": float(bankruptcies),
        "avg_ms": (
            sum(decision_times_ms) / len(decision_times_ms) if decision_times_ms else 0.0
        ),
        "max_ms": max(decision_times_ms) if decision_times_ms else 0.0,
    }
    return summary, team_ranks, team_caps


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ProfezIA across DME scenarios")
    parser.add_argument("--seeds", type=int, default=3, help="seeds per scenario (default: 3)")
    parser.add_argument("--agent", type=str, default="ProfezIA", help="target agent ID")
    parser.add_argument("--quick", action="store_true", help="run fewer/shorter standard scenarios")
    parser.add_argument(
        "--sparring",
        action="store_true",
        help="replace weak built-in opponents with competitive sparring bots",
    )
    parser.add_argument(
        "--private-stress",
        action="store_true",
        help="include 6 synthetic unseen private-scenario stress tests",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="print aggregate average rank and capital for every participating bot",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    scenarios = _build_standard_scenarios(repo_root, quick=args.quick)
    if args.private_stress:
        scenarios.extend(_build_private_stress_scenarios(repo_root))

    arena = "SPARRING ARENA" if args.sparring else "BUILTIN BASELINES"
    print(f"Evaluating {args.agent} [{arena}] ({args.seeds} seed(s)/scenario)")
    header = (
        f"{'Scenario':<18} {'Rank':>6} {'Capital':>11} {'BestRival':>11} "
        f"{'Unsold':>7} {'Raw':>6} {'Avg ms':>8} {'Max ms':>8} {'Bankr':>5}"
    )
    print(header)
    print("-" * len(header))

    all_ranks: list[float] = []
    worst_ms = 0.0
    agg_ranks: dict[str, list[float]] = {}
    agg_caps: dict[str, list[float]] = {}

    for name, cfg, path, roster_mode in scenarios:
        res, t_ranks, t_caps = _evaluate_scenario(
            cfg, path, args.seeds, args.agent, args.sparring, roster_mode
        )
        all_ranks.append(res["avg_rank"])
        worst_ms = max(worst_ms, res["max_ms"])
        for tid, vals in t_ranks.items():
            agg_ranks.setdefault(tid, []).extend(vals)
        for tid, vals in t_caps.items():
            agg_caps.setdefault(tid, []).extend(vals)

        print(
            f"{name:<18} {res['avg_rank']:6.2f} {res['avg_capital']:11.2f} "
            f"{res['avg_best_rival']:11.2f} {res['avg_unsold']:7.1f} "
            f"{res['avg_raw']:6.1f} {res['avg_ms']:8.2f} {res['max_ms']:8.2f} "
            f"{int(res['bankruptcies']):5d}"
        )

    print("-" * len(header))
    print(
        f"Overall avg rank: {sum(all_ranks) / len(all_ranks):.2f} | "
        f"Worst decision latency: {worst_ms:.2f} ms (limit: 100.00 ms)"
    )

    if args.leaderboard:
        print("\nAggregate Leaderboard (lower avg rank is better):")
        lb_header = f"{'Agent':<22} {'Avg Rank':>10} {'Avg Capital':>14} {'Matches':>8}"
        print(lb_header)
        print("-" * len(lb_header))
        ordered = sorted(
            agg_ranks.items(),
            key=lambda kv: (sum(kv[1]) / len(kv[1]), -(sum(agg_caps[kv[0]]) / len(agg_caps[kv[0]]))),
        )
        for tid, vals in ordered:
            avg_r = sum(vals) / len(vals)
            avg_c = sum(agg_caps[tid]) / len(agg_caps[tid])
            print(f"{tid:<22} {avg_r:10.2f} {avg_c:14.2f} {len(vals):8d}")


if __name__ == "__main__":
    main()
