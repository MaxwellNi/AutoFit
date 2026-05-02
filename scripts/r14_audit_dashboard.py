#!/usr/bin/env python3
"""
Round-14 一次性审计仪表盘 —— 所有"未彻底检查"的问题一次性落盘到 JSON + MD。
可重跑、无副作用、只读。
"""
import json, glob, os, sys, statistics
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_audit_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

audit = {"timestamp_cest": datetime.now().isoformat(), "checks": {}}
_METRIC_READ_ERRORS = []
_METRIC_READ_ERROR_KEYS = set()


def _load_json(path):
    try:
        return json.load(open(path))
    except Exception as exc:
        key = (str(path), type(exc).__name__, str(exc))
        if key not in _METRIC_READ_ERROR_KEYS:
            _METRIC_READ_ERROR_KEYS.add(key)
            _METRIC_READ_ERRORS.append({
                "path": str(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        return None


def _coverage_metric_files():
    return sorted(set(
        glob.glob(str(ROOT/"runs/benchmarks/r13patch_*v2*/metrics.json")) +
        glob.glob(str(ROOT/"runs/benchmarks/r14fcast_*/metrics.json")) +
        glob.glob(str(ROOT/"runs/benchmarks/r14mond_*/metrics.json")) +
        glob.glob(str(ROOT/"runs/benchmarks/r14stud_*/metrics.json"))
    ))

# ─── Check 1: 旧 benchmark 是否已含 PI coverage？───────────────────
def check_old_bench_coverage():
    files = sorted(glob.glob(str(ROOT/"runs/benchmarks/block3_phase9_fair/**/metrics.json"), recursive=True))
    total = 0; has_nccopo = 0; has_cov90 = 0; has_pi90 = 0
    key_union = set()
    for f in files:
        m = _load_json(f)
        if m is None: continue
        if isinstance(m, dict) and 'results' in m: m = m['results']
        if not isinstance(m, list): continue
        for r in m:
            if not isinstance(r, dict): continue
            total += 1
            key_union |= set(r.keys())
            if r.get('nccopo_coverage_90') is not None: has_nccopo += 1
            if r.get('coverage_90') is not None: has_cov90 += 1
            if r.get('pi_coverage_90') is not None: has_pi90 += 1
    return dict(n_files=len(files), n_records=total, has_nccopo_c90=has_nccopo,
                has_coverage_90=has_cov90, has_pi_coverage_90=has_pi90,
                coverage_keys_in_union=sorted([k for k in key_union if 'cov' in k.lower() or 'nccopo' in k.lower()]))

# ─── Check 2: 新 v2 benchmark c90 分布 ─────────────────────────────
def check_v2_coverage():
    files = _coverage_metric_files()
    recs = []
    mondrian_recs = []
    mondrian_deltas = []
    studentized_recs = []
    studentized_deltas = []
    cqr_recs = []
    cqr_deltas = []
    canonical_cqr_recs = []
    canonical_cqr_deltas = []
    gpd_recs = []
    gpd_deltas = []
    for f in files:
        run_name = Path(f).parent.name
        canonical_cqr_protocol = ("cqrrow" in run_name) or ("cqrgpd" in run_name)
        m = _load_json(f)
        if m is None: continue
        if isinstance(m, dict) and 'results' in m: m = m['results']
        for r in m:
            if isinstance(r, dict) and r.get('nccopo_coverage_90') is not None:
                recs.append(dict(
                    model=r.get('model_name') or r.get('model'),
                    horizon=r.get('horizon'),
                    category=r.get('category'),
                    mae=r.get('mae'),
                    c90=r.get('nccopo_coverage_90'),
                    fair=r.get('fairness_pass'),
                ))
            if isinstance(r, dict) and r.get('nccopo_coverage_90_mondrian') is not None:
                mondrian_recs.append(dict(
                    model=r.get('model_name') or r.get('model'),
                    horizon=r.get('horizon'),
                    category=r.get('category'),
                    mae=r.get('mae'),
                    c90_mondrian=r.get('nccopo_coverage_90_mondrian'),
                    fair=r.get('fairness_pass'),
                ))
                if r.get('nccopo_coverage_90') is not None:
                    mondrian_deltas.append(
                        float(r.get('nccopo_coverage_90_mondrian')) - float(r.get('nccopo_coverage_90'))
                    )
            if isinstance(r, dict) and r.get('nccopo_coverage_90_studentized') is not None:
                studentized_recs.append(dict(
                    model=r.get('model_name') or r.get('model'),
                    horizon=r.get('horizon'),
                    category=r.get('category'),
                    mae=r.get('mae'),
                    c90_studentized=r.get('nccopo_coverage_90_studentized'),
                    fair=r.get('fairness_pass'),
                ))
                if r.get('nccopo_coverage_90') is not None:
                    studentized_deltas.append(
                        float(r.get('nccopo_coverage_90_studentized')) - float(r.get('nccopo_coverage_90'))
                    )
            if isinstance(r, dict) and r.get('nccopo_coverage_90_cqr_lite') is not None:
                cqr_item = dict(
                    run=run_name,
                    model=r.get('model_name') or r.get('model'),
                    horizon=r.get('horizon'),
                    category=r.get('category'),
                    c90_cqr_lite=r.get('nccopo_coverage_90_cqr_lite'),
                    fair=r.get('fairness_pass'),
                )
                cqr_recs.append(cqr_item)
                if canonical_cqr_protocol:
                    canonical_cqr_recs.append(cqr_item)
                if r.get('nccopo_coverage_90') is not None:
                    delta = float(r.get('nccopo_coverage_90_cqr_lite')) - float(r.get('nccopo_coverage_90'))
                    cqr_deltas.append(delta)
                    if canonical_cqr_protocol:
                        canonical_cqr_deltas.append(delta)
            if isinstance(r, dict) and r.get('nccopo_coverage_90_gpd_evt') is not None:
                gpd_recs.append(dict(
                    model=r.get('model_name') or r.get('model'),
                    horizon=r.get('horizon'),
                    category=r.get('category'),
                    c90_gpd_evt=r.get('nccopo_coverage_90_gpd_evt'),
                    fair=r.get('fairness_pass'),
                ))
                if r.get('nccopo_coverage_90') is not None:
                    gpd_deltas.append(
                        float(r.get('nccopo_coverage_90_gpd_evt')) - float(r.get('nccopo_coverage_90'))
                    )
    n = len(recs)
    c90s = [r['c90'] for r in recs if r['c90'] is not None]
    mondrian_c90s = [r['c90_mondrian'] for r in mondrian_recs if r['c90_mondrian'] is not None]
    studentized_c90s = [r['c90_studentized'] for r in studentized_recs if r['c90_studentized'] is not None]
    cqr_c90s = [r['c90_cqr_lite'] for r in cqr_recs if r['c90_cqr_lite'] is not None]
    canonical_cqr_c90s = [r['c90_cqr_lite'] for r in canonical_cqr_recs if r['c90_cqr_lite'] is not None]
    gpd_c90s = [r['c90_gpd_evt'] for r in gpd_recs if r['c90_gpd_evt'] is not None]
    return dict(n_records=n, n_files=len(files),
                c90_mean=statistics.mean(c90s) if c90s else None,
                c90_min=min(c90s) if c90s else None,
                c90_max=max(c90s) if c90s else None,
                undershoot_severity=statistics.mean([0.9-c for c in c90s]) if c90s else None,
                mondrian_n_records=len(mondrian_recs),
                mondrian_c90_mean=statistics.mean(mondrian_c90s) if mondrian_c90s else None,
                mondrian_c90_min=min(mondrian_c90s) if mondrian_c90s else None,
                mondrian_c90_max=max(mondrian_c90s) if mondrian_c90s else None,
                mondrian_delta_mean=statistics.mean(mondrian_deltas) if mondrian_deltas else None,
                studentized_n_records=len(studentized_recs),
                studentized_c90_mean=statistics.mean(studentized_c90s) if studentized_c90s else None,
                studentized_c90_min=min(studentized_c90s) if studentized_c90s else None,
                studentized_c90_max=max(studentized_c90s) if studentized_c90s else None,
                studentized_delta_mean=statistics.mean(studentized_deltas) if studentized_deltas else None,
                cqr_lite_n_records=len(cqr_recs),
                cqr_lite_c90_mean=statistics.mean(cqr_c90s) if cqr_c90s else None,
                cqr_lite_c90_min=min(cqr_c90s) if cqr_c90s else None,
                cqr_lite_c90_max=max(cqr_c90s) if cqr_c90s else None,
                cqr_lite_delta_mean=statistics.mean(cqr_deltas) if cqr_deltas else None,
                canonical_cqr_lite_n_records=len(canonical_cqr_recs),
                canonical_cqr_lite_c90_mean=statistics.mean(canonical_cqr_c90s) if canonical_cqr_c90s else None,
                canonical_cqr_lite_c90_min=min(canonical_cqr_c90s) if canonical_cqr_c90s else None,
                canonical_cqr_lite_c90_max=max(canonical_cqr_c90s) if canonical_cqr_c90s else None,
                canonical_cqr_lite_delta_mean=statistics.mean(canonical_cqr_deltas) if canonical_cqr_deltas else None,
                canonical_cqr_lite_protocol="runs whose directory name contains cqrrow or cqrgpd; excludes source-scaling/embedding diagnostic probes from the CQR pass gate while retaining raw_all fields above",
                gpd_evt_n_records=len(gpd_recs),
                gpd_evt_c90_mean=statistics.mean(gpd_c90s) if gpd_c90s else None,
                gpd_evt_c90_min=min(gpd_c90s) if gpd_c90s else None,
                gpd_evt_c90_max=max(gpd_c90s) if gpd_c90s else None,
                gpd_evt_delta_mean=statistics.mean(gpd_deltas) if gpd_deltas else None,
                records=recs)

# ─── Check 3: fairness_pass 分布 ───────────────────────────────────
def check_fairness():
    files = sorted(glob.glob(str(ROOT/"runs/benchmarks/block3_phase9_fair/**/metrics.json"), recursive=True))
    total = 0; passed = 0; failed = 0; unknown = 0
    fail_models = {}
    for f in files:
        m = _load_json(f)
        if m is None: continue
        if isinstance(m, dict) and 'results' in m: m = m['results']
        if not isinstance(m, list): continue
        for r in m:
            if not isinstance(r, dict): continue
            total += 1
            fp = r.get('fairness_pass')
            if fp is True: passed += 1
            elif fp is False:
                failed += 1
                mn = r.get('model_name') or r.get('model') or '?'
                fail_models[mn] = fail_models.get(mn, 0) + 1
            else: unknown += 1
    return dict(total=total, passed=passed, failed=failed, unknown=unknown,
                pass_rate=passed/total if total else None,
                top_fail_models=sorted(fail_models.items(), key=lambda x: -x[1])[:10])

# ─── Check 4: T_pub external validation 状态 ───────────────────────
def check_tpub():
    out = {}
    for ds in ("mimic_iii", "fremtpl2", "m5", "bikeshare"):
        hits = glob.glob(str(ROOT/f"runs/**/*{ds}*"), recursive=True) + glob.glob(str(ROOT/f"runs/**/*{ds.upper()}*"), recursive=True)
        json_hits = sorted([p for p in hits if str(p).endswith(".json")], key=os.path.getmtime, reverse=True)
        other_hits = sorted([p for p in hits if not str(p).endswith(".json")], key=os.path.getmtime, reverse=True)
        out[ds] = dict(exists=bool(hits), artifacts=(json_hits + other_hits)[:6])
    return out

# ─── Check 5: NC-CoPo wired 条件 vs 未 wired 条件 ────────────────
def check_nccopo_wire_coverage():
    old_files = glob.glob(str(ROOT/"runs/benchmarks/block3_phase9_fair/**/metrics.json"), recursive=True)
    v2_files = _coverage_metric_files()
    old_records = 0
    wired_records = 0
    for f in old_files:
        m = _load_json(f)
        if m is None: continue
        if isinstance(m, dict) and 'results' in m: m = m['results']
        if not isinstance(m, list): continue
        for r in m:
            if not isinstance(r, dict): continue
            old_records += 1
    for f in v2_files:
        m = _load_json(f)
        if m is None: continue
        if isinstance(m, dict) and 'results' in m: m = m['results']
        if not isinstance(m, list): continue
        for r in m:
            if isinstance(r, dict) and r.get('nccopo_coverage_90') is not None:
                wired_records += 1
    total = old_records + wired_records
    return dict(
        old_files=len(old_files),
        v2_files=len(v2_files),
        old_records=old_records,
        wired_records=wired_records,
        ratio_with_c90=f"{wired_records} / {total} = {wired_records/total*100:.2f}%" if total else None,
    )

# ─── Check 6: y-shift 影响范围（P0a）──────────────────────────────
def check_yshift_impact():
    # 复用 5347357 结论
    return dict(
        legacy_kept=422491,
        p0a_h7_kept=416803,
        p0a_h30_kept=397883,
        extra_drop_h7=39085,
        extra_drop_h7_ratio="9.3% of legacy-kept",
        extra_drop_h30=130722,
        extra_drop_h30_ratio="30.9% of legacy-kept",
        source_jobid=5347357,
    )

# ─── Check 7: iid sanity artifact ────────────────────────────────
def check_iid_sanity():
    sp = ROOT/"scripts/nccopo_iid_sanity.py"
    return dict(script_exists=sp.exists(),
                last_result_manual="7/7 PASS (normal/t3/pareto, cov=0.90±0.01)",
                path=str(sp.relative_to(ROOT)))

# ─── Check 8: task 覆盖（哪些 task×ablation×model 还没 c90） ──────
def check_coverage_matrix():
    files = _coverage_metric_files()
    have = set()
    have_mondrian = set()
    have_studentized = set()
    have_cqr = set()
    have_gpd = set()
    for f in files:
        m = _load_json(f)
        if m is None: continue
        if isinstance(m, dict) and 'results' in m: m = m['results']
        for r in m:
            if not isinstance(r, dict): continue
            if r.get('nccopo_coverage_90') is not None:
                key = (r.get('task') or '?', r.get('ablation') or '?',
                       r.get('category') or '?', r.get('horizon'))
                have.add(key)
            if r.get('nccopo_coverage_90_mondrian') is not None:
                key = (r.get('task') or '?', r.get('ablation') or '?',
                       r.get('category') or '?', r.get('horizon'))
                have_mondrian.add(key)
            if r.get('nccopo_coverage_90_studentized') is not None:
                key = (r.get('task') or '?', r.get('ablation') or '?',
                       r.get('category') or '?', r.get('horizon'))
                have_studentized.add(key)
            if r.get('nccopo_coverage_90_cqr_lite') is not None:
                key = (r.get('task') or '?', r.get('ablation') or '?',
                       r.get('category') or '?', r.get('horizon'))
                have_cqr.add(key)
            if r.get('nccopo_coverage_90_gpd_evt') is not None:
                key = (r.get('task') or '?', r.get('ablation') or '?',
                       r.get('category') or '?', r.get('horizon'))
                have_gpd.add(key)
    return dict(
        n_cells_with_c90=len(have),
        cells=sorted(have, key=str),
        n_cells_with_mondrian_c90=len(have_mondrian),
        mondrian_cells=sorted(have_mondrian, key=str),
        n_cells_with_studentized_c90=len(have_studentized),
        studentized_cells=sorted(have_studentized, key=str),
        n_cells_with_cqr_lite_c90=len(have_cqr),
        cqr_lite_cells=sorted(have_cqr, key=str),
        n_cells_with_gpd_evt_c90=len(have_gpd),
        gpd_evt_cells=sorted(have_gpd, key=str),
    )

def check_metric_read_errors():
    return dict(
        n_errors=len(_METRIC_READ_ERRORS),
        errors=_METRIC_READ_ERRORS[:50],
    )

audit["checks"]["1_old_benchmark_coverage"] = check_old_bench_coverage()
audit["checks"]["2_v2_coverage"] = check_v2_coverage()
audit["checks"]["3_fairness"] = check_fairness()
audit["checks"]["4_tpub_external"] = check_tpub()
audit["checks"]["5_nccopo_wire_ratio"] = check_nccopo_wire_coverage()
audit["checks"]["6_yshift_impact"] = check_yshift_impact()
audit["checks"]["7_iid_sanity"] = check_iid_sanity()
audit["checks"]["8_coverage_matrix"] = check_coverage_matrix()
audit["checks"]["9_metric_read_errors"] = check_metric_read_errors()

with open(OUT_JSON, "w") as f: json.dump(audit, f, indent=2, default=str)

# MD report
lines = [f"# R14 Audit Dashboard — {audit['timestamp_cest']}", ""]
for name, c in audit["checks"].items():
    lines.append(f"## {name}")
    lines.append("```json")
    lines.append(json.dumps(c, indent=2, default=str)[:2500])
    lines.append("```")
    lines.append("")
with open(OUT_MD, "w") as f: f.write("\n".join(lines))

print(f"OK: {OUT_JSON}")
print(f"OK: {OUT_MD}")
print()
for name, c in audit["checks"].items():
    print(f"─── {name} ───")
    if isinstance(c, dict):
        for k, v in c.items():
            s = str(v)
            if len(s) > 200: s = s[:200] + "..."
            print(f"  {k}: {s}")
    print()
