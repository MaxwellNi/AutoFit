#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(cat "${ROOT}/runs/backups/current_audit_stamp.txt")"
OUT="${ROOT}/runs/sanity_${STAMP}"
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"

FEAS_JSON="${1:-}"
OFFERS_CORE="${2:-}"

if [ -z "${FEAS_JSON}" ] || [ -z "${OFFERS_CORE}" ]; then
  echo "usage: $0 <feasibility_json> <offers_core_parquet>" >&2
  exit 1
fi
if [ ! -f "${FEAS_JSON}" ]; then
  echo "missing feasibility json: ${FEAS_JSON}" >&2
  exit 1
fi
if [ ! -f "${OFFERS_CORE}" ]; then
  echo "missing offers_core: ${OFFERS_CORE}" >&2
  exit 1
fi

HORIZON="$(
python - "${FEAS_JSON}" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
rec = data.get("recommendation", {})
print(rec.get("recommended_horizon"))
PY
)"

if [ -z "${HORIZON}" ] || [ "${HORIZON}" = "None" ]; then
  echo "recommended_horizon missing in ${FEAS_JSON}" >&2
  exit 1
fi

echo "recommended_horizon=${HORIZON}" | tee "${LOGS}/ab_horizon.txt"

EXP0="sf_ab_real_edgar_off_lr100000_goal50_h${HORIZON}_sf0"
EXP1="sf_ab_real_edgar_off_lr100000_goal50_h${HORIZON}_sf1"

PYTHONPATH=src python "${ROOT}/scripts/run_full_benchmark.py" \
  --offers_core "${OFFERS_CORE}" \
  --use_edgar 0 \
  --limit_rows 100000 \
  --exp_name "${EXP0}" \
  --plan paper_min \
  --strict_matrix 1 \
  --models dlinear \
  --fusion_types none \
  --module_variants base \
  --seeds 42 \
  --sample_strategy random_entities \
  --sample_seed 42 \
  --split_seed 42 \
  --label_goal_min 50 \
  --label_horizon "${HORIZON}" \
  --strict_future 0 \
  --min_label_delta_days 1.0 \
  --min_ratio_delta_rel 1e-4 \
  |& tee "${LOGS}/${EXP0}.log"

PYTHONPATH=src python "${ROOT}/scripts/run_full_benchmark.py" \
  --offers_core "${OFFERS_CORE}" \
  --use_edgar 0 \
  --limit_rows 100000 \
  --exp_name "${EXP1}" \
  --plan paper_min \
  --strict_matrix 1 \
  --models dlinear \
  --fusion_types none \
  --module_variants base \
  --seeds 42 \
  --sample_strategy random_entities \
  --sample_seed 42 \
  --split_seed 42 \
  --label_goal_min 50 \
  --label_horizon "${HORIZON}" \
  --strict_future 1 \
  --min_label_delta_days 1.0 \
  --min_ratio_delta_rel 1e-4 \
  |& tee "${LOGS}/${EXP1}.log"

ROOT="${ROOT}" python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["ROOT"])
stamp = (root / "runs" / "backups" / "current_audit_stamp.txt").read_text().strip()
bench_root = root / "runs" / "benchmarks"
out = root / "runs" / f"sanity_{stamp}"

bench_list = out / "bench_dirs_ab_real.txt"
def pick(tag: str) -> Path:
    paths = [p for p in bench_root.glob(f"sf_ab_real_edgar_off_lr100000_goal50_h*_{tag}_*") if p.is_dir()]
    if not paths:
        raise SystemExit(f"missing bench_dir for tag={tag}")
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]

sf0 = pick("sf0")
sf1 = pick("sf1")

bench_list.write_text("\n".join([str(sf0), str(sf1)]) + "\n", encoding="utf-8")
print(bench_list)
PY

bench_dirs="$(tr '\n' ' ' < "${OUT}/bench_dirs_ab_real.txt")"

PYTHONPATH=src python "${ROOT}/scripts/audit_bench_configs.py" \
  --bench_dirs ${bench_dirs} \
  --output_path "${OUT}/audit_summary_ab_real.json" \
  |& tee "${LOGS}/gateA_ab_real.log"

sf0_dir="$(head -n 1 "${OUT}/bench_dirs_ab_real.txt")"
sf1_dir="$(tail -n 1 "${OUT}/bench_dirs_ab_real.txt")"

PYTHONPATH=src python "${ROOT}/scripts/check_label_leakage.py" \
  --offers_core "${OFFERS_CORE}" \
  --use_edgar 0 \
  --limit_rows 100000 \
  --sample_strategy random_entities \
  --sample_seed 42 \
  --split_seed 42 \
  --label_goal_min 50 \
  --label_horizon "${HORIZON}" \
  --seeds 42 \
  --strict_future 0 \
  --min_label_delta_days 1.0 \
  --min_ratio_delta_rel 1e-4 \
  --output_dir "${OUT}_leakage_ab_real_sf0" \
  --bench_dir "${sf0_dir}" \
  --exp_name "${EXP0}" \
  |& tee "${LOGS}/gateB_ab_real_sf0.log"

PYTHONPATH=src python "${ROOT}/scripts/check_label_leakage.py" \
  --offers_core "${OFFERS_CORE}" \
  --use_edgar 0 \
  --limit_rows 100000 \
  --sample_strategy random_entities \
  --sample_seed 42 \
  --split_seed 42 \
  --label_goal_min 50 \
  --label_horizon "${HORIZON}" \
  --seeds 42 \
  --strict_future 1 \
  --min_label_delta_days 1.0 \
  --min_ratio_delta_rel 1e-4 \
  --output_dir "${OUT}_leakage_ab_real_sf1" \
  --bench_dir "${sf1_dir}" \
  --exp_name "${EXP1}" \
  |& tee "${LOGS}/gateB_ab_real_sf1.log"

PYTHONPATH=src python "${ROOT}/scripts/sanity_check_metrics.py" \
  --bench_dirs ${bench_dirs} \
  --output_dir "${OUT}" \
  |& tee "${LOGS}/gateC_ab_real.log"

ROOT="${ROOT}" python - <<'PY'
import os
from pathlib import Path
import json
root = Path(os.environ["ROOT"])
stamp = (root / "runs" / "backups" / "current_audit_stamp.txt").read_text().strip()
out = root / "runs" / f"sanity_{stamp}"
path = out / "sanity_report.json"
if path.exists():
    data = json.loads(path.read_text())
    (out / "sanity_report_ab_real.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
PY

PYTHONPATH=src python "${ROOT}/scripts/audit_progress_alignment.py" \
  --bench_dirs ${bench_dirs} \
  --output_path "${OUT}/alignment_audit_ab_real.json" \
  --min_label_delta_days 1.0 \
  |& tee "${LOGS}/gateD_ab_real.log"

PYTHONPATH=src python "${ROOT}/scripts/summarize_horizon_audit.py" \
  --sanity_report "${OUT}/sanity_report_ab_real.json" \
  --alignment_audit "${OUT}/alignment_audit_ab_real.json" \
  --output_dir "${OUT}" \
  |& tee "${LOGS}/summarize_ab_real.log"

ROOT="${ROOT}" python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["ROOT"])
stamp = (root / "runs" / "backups" / "current_audit_stamp.txt").read_text().strip()
out = root / "runs" / f"sanity_{stamp}"
for name in ["horizon_compare.json", "horizon_compare.md"]:
    src = out / name
    if src.exists():
        dst = out / name.replace("horizon_compare", "horizon_compare_ab_real")
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
PY

ROOT="${ROOT}" python - <<'PY'
import os
from pathlib import Path
import json
import yaml

root = Path(os.environ["ROOT"])
stamp = (root / "runs" / "backups" / "current_audit_stamp.txt").read_text().strip()
out = root / "runs" / f"sanity_{stamp}"

bench_dirs = [l.strip() for l in (out / "bench_dirs_ab_real.txt").read_text().splitlines() if l.strip()]
if len(bench_dirs) != 2:
    raise SystemExit("expected 2 bench dirs")

sf_entries = []
for idx, bench in enumerate(bench_dirs):
    bench_path = Path(bench)
    cfg_path = bench_path / "configs" / "resolved_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    sanity = json.loads((out / "sanity_report_ab_real.json").read_text())
    sanity_entry = next((b for b in sanity.get("benchmarks", []) if b.get("bench_dir") == str(bench_path)), {})
    n_eval = None
    if sanity_entry.get("naive_progress_baseline"):
        n_eval = sanity_entry["naive_progress_baseline"].get("n_eval")
    y_count = sanity_entry.get("y_stats", {}).get("count")
    sanity_errors = sanity_entry.get("errors")

    align = json.loads((out / "alignment_audit_ab_real.json").read_text())
    align_entry = next((r for r in align.get("runs", []) if r.get("bench_dir") == str(bench_path)), {})
    conclusion = align_entry.get("conclusion")
    n_rows = align_entry.get("n_rows")
    align_errors = align_entry.get("errors")

    leakage_dir = root / f"runs/sanity_{stamp}_leakage_ab_real_sf{idx}"
    leakage_flag = None
    if (leakage_dir / "label_leakage_report.json").exists():
        leakage = json.loads((leakage_dir / "label_leakage_report.json").read_text())
        leakage_flag = leakage.get("label_vs_current_ratio", {}).get("leakage_flag")

    sf_entries.append(
        {
            "bench_dir": str(bench_path),
            "strict_future": cfg.get("strict_future"),
            "label_horizon": cfg.get("label_horizon"),
            "n_rows": n_rows,
            "n_eval": n_eval,
            "y_count": y_count,
            "dropped_due_to_insufficient_future": cfg.get("dropped_due_to_insufficient_future"),
            "alignment_conclusion": conclusion,
            "leakage_flag": leakage_flag,
            "sanity_errors": sanity_errors,
            "alignment_errors": align_errors,
        }
    )

changed = any(sf_entries[0].get(k) != sf_entries[1].get(k) for k in (
    "n_rows", "n_eval", "dropped_due_to_insufficient_future", "alignment_conclusion", "leakage_flag"
))

lines_md = [
    "# strict_future AB (real offers_core_v2) result\\n",
    "| strict_future | n_rows | n_eval | y_count | dropped_due_to_insufficient_future | alignment_conclusion | leakage_flag | errors | bench_dir |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for entry in sf_entries:
    errors = []
    if entry.get("sanity_errors"):
        errors.append(f"sanity={entry['sanity_errors']}")
    if entry.get("alignment_errors"):
        errors.append(f"alignment={entry['alignment_errors']}")
    err_txt = "; ".join(errors) if errors else ""
    lines_md.append(
        f"| {entry['strict_future']} | {entry.get('n_rows')} | {entry.get('n_eval')} | {entry.get('y_count')} | {entry.get('dropped_due_to_insufficient_future')} | {entry.get('alignment_conclusion')} | {entry.get('leakage_flag')} | {err_txt} | {entry['bench_dir']} |"
    )

lines_md.append("\\n## Conclusion\\n")
if any(entry.get("sanity_errors") for entry in sf_entries):
    lines_md.append("AB 不完整：存在无预测/无样本导致的 sanity/alignment 错误，strict_future=1 未完成。")
elif changed:
    lines_md.append("strict_future 开关产生了可观测差异（样本量/对齐/泄露至少一项变化）。")
else:
    lines_md.append("strict_future 开关未产生可观测差异，需要检查接线或记录字段。")

(out / "RESULT_AB_REAL.md").write_text("\\n".join(lines_md) + "\\n", encoding="utf-8")
print(out / "RESULT_AB_REAL.md")
PY
