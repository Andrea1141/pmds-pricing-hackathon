"""Launch the DME graphical window (or terminal UI) with ProfezIA vs sparring bots.

Examples:
    uv run python scripts/watch_sparring.py --turns 40 --turn-delay 0.5
    uv run python scripts/watch_sparring.py --config config/scaled_market.yaml --turn-delay 0.4
    uv run python scripts/watch_sparring.py --include-builtins --turns 30 --turn-delay 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from sparring_opponents import build_sparring_roster  # noqa: E402

from dme_simulator.agents.loader import (  # noqa: E402
    load_simulation_agents,
    resolve_agent_directories,
)
from dme_simulator.config import load_config  # noqa: E402
from dme_simulator.runner import run_simulation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch ProfezIA compete against sparring bots in the DME UI"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="YAML scenario path (default: config/default.yaml)",
    )
    parser.add_argument("--turns", type=int, default=None, help="override number of turns")
    parser.add_argument("--seed", type=int, default=None, help="override random seed")
    parser.add_argument(
        "--ui",
        choices=("window", "terminal", "none"),
        default="window",
        help="dashboard UI mode (default: window)",
    )
    parser.add_argument(
        "--turn-delay",
        type=float,
        default=0.5,
        help="seconds per turn in live UI (default: 0.5)",
    )
    parser.add_argument(
        "--include-builtins",
        action="store_true",
        help="also include DummyAgent, PatientAgent, and PriceWarAgent (8-team room)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="ProfezIA",
        help="your custom agent ID (default: ProfezIA)",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    directories = resolve_agent_directories(config_path, config.agents.directory, None)
    discovered = load_simulation_agents(directories, config.agents.include)
    if args.agent not in discovered:
        raise SystemExit(f"Agent {args.agent!r} not found in {list(discovered)}")

    if args.include_builtins:
        agents = dict(discovered)
        agents.update(build_sparring_roster())
    else:
        agents = build_sparring_roster()
        agents[args.agent] = discovered[args.agent]

    run_simulation(
        config,
        agents,
        seed_override=args.seed,
        turns_override=args.turns,
        dashboard_override=args.ui,
        turn_delay_override=args.turn_delay,
    )


if __name__ == "__main__":
    main()
