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

# ─── Check 1: 旧 benchmark 是否已含 PI coverage？───────────────────
def check_old_bench_coverage():
    files = sorted(glob.glob(str(ROOT/"runs/benchmarks/block3_phase9_fair/**/metrics.json"), recursive=True))
    total = 0; has_nccopo = 0; has_cov90 = 0; has_pi90 = 0
    key_union = set()
    for f in files:
        try: m = json.load(open(f))
        except: continue
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
    files = sorted(set(
        glob.glob(str(ROOT/"runs/benchmarks/r13patch_*v2*/metrics.json")) +
        glob.glob(str(ROOT/"runs/benchmarks/r14fcast_*/metrics.json"))
    ))
    recs = []
    for f in files:
        try: m = json.load(open(f))
        except: continue
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
    n = len(recs)
    c90s = [r['c90'] for r in recs if r['c90'] is not None]
    return dict(n_records=n, n_files=len(files),
                c90_mean=statistics.mean(c90s) if c90s else None,
                c90_min=min(c90s) if c90s else None,
                c90_max=max(c90s) if c90s else None,
                undershoot_severity=statistics.mean([0.9-c for c in c90s]) if c90s else None,
                records=recs)

# ─── Check 3: fairness_pass 分布 ───────────────────────────────────
def check_fairness():
    files = sorted(glob.glob(str(ROOT/"runs/benchmarks/block3_phase9_fair/**/metrics.json"), recursive=True))
    total = 0; passed = 0; failed = 0; unknown = 0
    fail_models = {}
    for f in files:
        try: m = json.load(open(f))
        except: continue
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
    for ds in ("mimic_iii", "fremtpl2", "m5"):
        hits = glob.glob(str(ROOT/f"runs/**/*{ds}*"), recursive=True) + glob.glob(str(ROOT/f"runs/**/*{ds.upper()}*"), recursive=True)
        out[ds] = dict(exists=bool(hits), artifacts=hits[:3])
    return out

# ─── Check 5: NC-CoPo wired 条件 vs 未 wired 条件 ────────────────
def check_nccopo_wire_coverage():
    # 旧 benchmark 全是 NC-CoPo wire 未启用；v2 全启用
    # 计算：旧 17159 条 = 100% 无 c90；v2 ~20 条有 c90
    old_files = glob.glob(str(ROOT/"runs/benchmarks/block3_phase9_fair/**/metrics.json"), recursive=True)
    v2_files = glob.glob(str(ROOT/"runs/benchmarks/r13patch_*v2*/metrics.json"))
    return dict(old_files=len(old_files), v2_files=len(v2_files),
                ratio_with_c90=f"~20 / {17159+20} = {20/17179*100:.2f}%")

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
    files = sorted(set(
        glob.glob(str(ROOT/"runs/benchmarks/r13patch_*v2*/metrics.json")) +
        glob.glob(str(ROOT/"runs/benchmarks/r14fcast_*/metrics.json"))
    ))
    have = set()
    for f in files:
        try: m = json.load(open(f))
        except: continue
        if isinstance(m, dict) and 'results' in m: m = m['results']
        for r in m:
            if not isinstance(r, dict): continue
            if r.get('nccopo_coverage_90') is not None:
                key = (r.get('task') or '?', r.get('ablation') or '?',
                       r.get('category') or '?', r.get('horizon'))
                have.add(key)
    return dict(n_cells_with_c90=len(have), cells=sorted(have, key=str))

audit["checks"]["1_old_benchmark_coverage"] = check_old_bench_coverage()
audit["checks"]["2_v2_coverage"] = check_v2_coverage()
audit["checks"]["3_fairness"] = check_fairness()
audit["checks"]["4_tpub_external"] = check_tpub()
audit["checks"]["5_nccopo_wire_ratio"] = check_nccopo_wire_coverage()
audit["checks"]["6_yshift_impact"] = check_yshift_impact()
audit["checks"]["7_iid_sanity"] = check_iid_sanity()
audit["checks"]["8_coverage_matrix"] = check_coverage_matrix()

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
