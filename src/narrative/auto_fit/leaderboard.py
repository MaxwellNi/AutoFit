from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import yaml

from narrative.auto_fit.budget_search import SearchResult


def write_leaderboard(results: List[SearchResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {"candidate": r.candidate, "val_loss": r.val_loss, "epochs": r.epochs}
        for r in results
    ]
    leaderboard_path = out_dir / "leaderboard.json"
    leaderboard_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    best = min(results, key=lambda r: r.val_loss)
    best_path = out_dir / "best_config.yaml"
    best_path.write_text(yaml.safe_dump(best.candidate, sort_keys=False), encoding="utf-8")
    return best_path


__all__ = ["write_leaderboard"]
