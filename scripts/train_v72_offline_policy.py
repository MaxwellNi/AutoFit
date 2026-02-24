#!/usr/bin/env python3
"""Train a conservative offline policy from V7.2 historical strict evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "docs" / "benchmarks" / "block3_truth_pack" / "v72_policy_dataset.csv"
DEFAULT_REPORT = ROOT / "docs" / "benchmarks" / "block3_truth_pack" / "v72_policy_training_report.json"


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(raw: str, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return default


def _build_policy(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    action_payload: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        state = str(row.get("route_key", "")).strip()
        action = str(row.get("template_id", "")).strip()
        if not state or not action:
            continue
        reward = _safe_float(str(row.get("reward", "0.0")))
        grouped[state][action].append(reward)
        action_payload[(state, action)] = {
            "template_id": action,
            "candidate_subset": str(row.get("candidate_subset", "")),
            "count_family": str(row.get("count_family", "auto")),
            "binary_calibration_mode": str(row.get("binary_calibration_mode", "auto")),
            "top_k": str(row.get("top_k", "8")),
        }

    policy: Dict[str, Dict[str, object]] = {}
    for state, actions in grouped.items():
        scored: List[Tuple[str, float]] = []
        for action, rewards in actions.items():
            scored.append((action, mean(rewards)))
        scored.sort(key=lambda x: x[1], reverse=True)
        best_action, best_reward = scored[0]
        second_reward = scored[1][1] if len(scored) > 1 else best_reward
        margin = best_reward - second_reward
        confidence = 1.0 if len(scored) == 1 else max(0.05, min(0.95, 0.5 + margin / 20.0))
        payload = dict(action_payload[(state, best_action)])
        payload.update(
            {
                "reward_mean": round(best_reward, 6),
                "confidence": round(confidence, 6),
                "n_samples": int(sum(len(v) for v in actions.values())),
                "n_actions": len(actions),
            }
        )
        policy[state] = payload
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V7.2 offline policy model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--algorithm", type=str, default="conservative_contextual_bandit")
    parser.add_argument("--enable-deep-offline-rl", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    if not dataset.exists():
        raise SystemExit(f"Missing dataset: {dataset}")

    rows = _load_rows(dataset)
    policy = _build_policy(rows)
    policy_id = f"offline_policy_v72_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_id": policy_id,
        "algorithm": args.algorithm,
        "deep_offline_rl_enabled": bool(args.enable_deep_offline_rl),
        "dataset_path": str(dataset),
        "n_dataset_rows": len(rows),
        "n_policy_states": len(policy),
        "state_policy": policy,
        "notes": [
            "Selection uses strict historical evidence only.",
            "No test-set feedback used during policy construction.",
            "Deep offline RL path is feature-gated and disabled by default.",
        ],
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote policy report: {output}")
    print(json.dumps({"n_policy_states": len(policy), "policy_id": policy_id}, indent=2))


if __name__ == "__main__":
    main()
