"""Multi-scenario evaluation and profiling harness for ProfezIA.

Run from the repository root with:
    uv run python scripts/eval_profezia.py
    uv run python scripts/eval_profezia.py --seeds 5
    uv run python scripts/eval_profezia.py --quick
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dme_simulator.agents.loader import (
    load_simulation_agents,
    resolve_agent_directories,
)
from dme_simulator.config import EnvironmentConfig, load_config
from dme_simulator.runner import run_simulation


def _build_scenarios(repo_root: Path, quick: bool) -> list[tuple[str, EnvironmentConfig, Path]]:
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
        ("default", default_cfg, default_path),
        ("scaled_market", scaled_cfg, scaled_path),
        ("stress_cap4", cap4_cfg, default_path),
        ("stress_obs_all", obs_all_cfg, default_path),
        ("stress_short12", short12_cfg, default_path),
        ("tutorial_auction", tut_auc_cfg, tut_auc_path),
        ("tutorial_pricing", tut_prc_cfg, tut_prc_path),
    ]
    if quick:
        return [
            (name, replace(cfg, turns=min(cfg.turns, 15)), path)
            for name, cfg, path in scenarios[:4]
        ]
    return scenarios


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
    name: str,
    config: EnvironmentConfig,
    config_path: Path,
    seeds: int,
    target_agent: str,
) -> dict[str, float]:
    quiet_cfg = replace(
        config,
        output=replace(
            config.output,
            render_plots=False,
            progress_interval_turns=0,
            dashboard="none",
        ),
    )
    directories = resolve_agent_directories(config_path, quiet_cfg.agents.directory, None)

    ranks: list[float] = []
    capitals: list[float] = []
    unsold_goods: list[int] = []
    raw_leftover: list[int] = []
    decision_times_ms: list[float] = []
    bankruptcies = 0
    best_rival_caps: list[float] = []

    for offset in range(seeds):
        seed = quiet_cfg.seed + offset
        agents = load_simulation_agents(directories, quiet_cfg.agents.include)
        if target_agent not in agents:
            raise SystemExit(f"Target agent {target_agent!r} not found in {list(agents)}")

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

    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ProfezIA across DME scenarios")
    parser.add_argument("--seeds", type=int, default=3, help="seeds per scenario (default: 3)")
    parser.add_argument("--agent", type=str, default="ProfezIA", help="target agent ID")
    parser.add_argument("--quick", action="store_true", help="run fewer/shorter scenarios")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    scenarios = _build_scenarios(repo_root, quick=args.quick)

    print(f"Evaluating {args.agent} ({args.seeds} seed(s)/scenario)")
    header = (
        f"{'Scenario':<18} {'Rank':>6} {'Capital':>11} {'BestRival':>11} "
        f"{'Unsold':>7} {'Raw':>6} {'Avg ms':>8} {'Max ms':>8} {'Bankr':>5}"
    )
    print(header)
    print("-" * len(header))

    all_ranks: list[float] = []
    worst_ms = 0.0
    for name, cfg, path in scenarios:
        res = _evaluate_scenario(name, cfg, path, args.seeds, args.agent)
        all_ranks.append(res["avg_rank"])
        worst_ms = max(worst_ms, res["max_ms"])
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


if __name__ == "__main__":
    main()
