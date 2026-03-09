#!/usr/bin/env python3
"""Generate Phase 9 Wave 3 SLURM scripts.

Wave 3 covers:
  1. MEMORY-FIXED resubmits for _ct/_fu OOM failures on bigmem
  2. l40s partition exploitation (iris-snt QOS) for GPU overflow
  3. Gap-fill for any remaining models

Memory allocation based on EMPIRICAL MaxRSS from completed Wave 1 jobs:
  - core_only:  ~120GB peak → request 256G (safe margin)
  - core_edgar: ~175GB peak → request 256G
  - core_text:  ~283GB peak → request 512G
  - full:       ~353GB peak → request 640G

Partition strategy:
  - GPU models: gpu partition (756G RAM, 4x V100) — primary
  - GPU models: l40s partition (515G RAM, 4x L40S) — overflow via iris-snt QOS
  - CPU models (statistical, ml_tabular, autofit): bigmem (3TB RAM)
"""
from pathlib import Path
import stat, textwrap

OUT_DIR = Path(__file__).resolve().parent.parent / ".slurm_scripts" / "phase9"
LOG_BASE = "/work/projects/eint/logs/phase9"

# ── Memory tiers based on empirical MaxRSS ───────────────────────
MEM_BY_ABLATION_BIGMEM = {
    "core_only": "256G",
    "core_edgar": "256G",
    "core_text": "512G",
    "full": "640G",
}

MEM_BY_ABLATION_GPU = {
    "core_only": "256G",
    "core_edgar": "256G",
    "core_text": "400G",
    "full": "500G",
}

MEM_BY_ABLATION_L40S = {
    "core_only": "200G",
    "core_edgar": "200G",
    "core_text": "400G",
    "full": "500G",
}

# ── Task / ablation grid ────────────────────────────────────────────
TASK_ABLATIONS = {
    "task1_outcome": ["core_only", "core_edgar", "core_text", "full"],
    "task2_forecast": ["core_only", "core_edgar", "core_text", "full"],
    "task3_risk_adjust": ["core_only", "core_edgar", "full"],
}
ABL_SHORT = {"core_only": "co", "core_edgar": "ce", "core_text": "ct", "full": "fu"}
TASK_SHORT = {"task1_outcome": "t1", "task2_forecast": "t2", "task3_risk_adjust": "t3"}

# ── Account templates ──────────────────────────────────────────────
def _npin_env_block() -> str:
    return textwrap.dedent("""\
        export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
        eval "$(micromamba shell hook -s bash)"
        micromamba activate insider
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        cd /home/users/npin/repo_root
        INSIDER_PY="${CONDA_PREFIX}/bin/python3"
    """)

def _cfisch_env_block() -> str:
    return textwrap.dedent("""\
        INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
        export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
        cd /work/projects/eint/repo_root
    """)


def gen_script(
    name: str,
    account: str,          # "npin" | "cfisch"
    partition: str,        # batch | gpu | bigmem | l40s
    mem: str,              # e.g. "256G"
    cpus: int,
    gpu: bool,
    task: str,
    ablation: str,
    category: str,
    models: str,
    preset: str = "full",
    output_dir_override: str | None = None,
    qos: str = "normal",
) -> str:
    acct = "christian.fisch" if account == "cfisch" else "npin"
    env_block = _cfisch_env_block() if account == "cfisch" else _npin_env_block()
    gres_line = "#SBATCH --gres=gpu:1\n" if gpu else ""
    odir = output_dir_override or f"runs/benchmarks/block3_phase9_fair/{task}/{category}/{ablation}"

    script = f"""#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --account={acct}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --time=2-00:00:00
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
{gres_line}#SBATCH --output={LOG_BASE}/{name}_%j.out
#SBATCH --error={LOG_BASE}/{name}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
{env_block}
if [[ ! -x "${{INSIDER_PY}}" ]]; then
  echo "FATAL: insider python missing: ${{INSIDER_PY}}"; exit 2
fi
echo "============================================================"
echo "Phase 9 Fair Benchmark | Job ${{SLURM_JOB_ID}} on $(hostname)"
echo "$(date -Iseconds) | Python: $(${{INSIDER_PY}} -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

echo "Task: {task} | Cat: {category} | Abl: {ablation} | Models: {models}"
"${{INSIDER_PY}}" scripts/run_block3_benchmark_shard.py \\
    --task {task} --category {category} --ablation {ablation} \\
    --preset {preset} --output-dir {odir} --seed 42 \\
    --no-verify-first --models {models}
echo "Done: $(date -Iseconds)"
"""
    return script.lstrip()


def write_script(name: str, content: str):
    path = OUT_DIR / f"{name}.sh"
    prev_blank = False
    clean = []
    for line in content.splitlines():
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        clean.append(line)
        prev_blank = is_blank
    path.write_text("\n".join(clean) + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def make_full_combos(prefix, account, partition, mem_map, cpus, gpu, category, models,
                     only_combos=None, qos="normal"):
    """Generate scripts for all task/ablation combos with per-ablation memory."""
    scripts = []
    for task, ablations in TASK_ABLATIONS.items():
        for abl in ablations:
            ts = TASK_SHORT[task]
            als = ABL_SHORT[abl]
            combo = f"{ts}_{als}"
            if only_combos and combo not in only_combos:
                continue
            name = f"{prefix}_{combo}"
            # Use per-ablation memory from the map
            mem = mem_map[abl] if isinstance(mem_map, dict) else mem_map
            content = gen_script(
                name=name, account=account, partition=partition,
                mem=mem, cpus=cpus, gpu=gpu, qos=qos,
                task=task, ablation=abl, category=category, models=models,
            )
            write_script(name, content)
            scripts.append(name)
    return scripts


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_scripts = {"npin": [], "cfisch": []}

    # ═══════════════════════════════════════════════════════════════════
    # 1. MEMORY-FIXED BIGMEM RESUBMITS (w3r_*) — cfisch account
    # ═══════════════════════════════════════════════════════════════════

    # Statistical (10 new models): ALL ablations OOM'd at 128G
    # _co/_ce need 256G, _ct needs 512G, _fu needs 640G
    STAT_NEW = "AutoCES,CrostonClassic,CrostonOptimized,CrostonSBA,DynamicOptimizedTheta,HistoricAverage,Holt,HoltWinters,Naive,WindowAverage"
    s = make_full_combos("w3r_sta", "cfisch", "bigmem", MEM_BY_ABLATION_BIGMEM,
                         28, False, "statistical", STAT_NEW)
    all_scripts["cfisch"].extend(s)

    # AutoFit V736: _ct/_fu OOM'd at 256G, _ce survived at 256G
    # Only resubmit _ct and _fu with higher memory
    af9_oom = {"t1_ct", "t1_fu", "t2_ct", "t2_fu", "t3_fu"}
    s = make_full_combos("w3r_af9", "cfisch", "bigmem", MEM_BY_ABLATION_BIGMEM,
                         28, False, "autofit", "AutoFitV736", only_combos=af9_oom)
    all_scripts["cfisch"].extend(s)

    # ML Tabular: _ct/_fu likely to OOM at 256G (based on pattern)
    # Pre-emptively generate with correct memory for when current jobs fail
    MLT_MODELS = "LightGBM,XGBoost,CatBoost,RandomForest,ExtraTrees,HistGradientBoosting,Ridge,Lasso,ElasticNet,SVR,KNN,QuantileRegressor,MeanPredictor,SeasonalNaive,LightGBMTweedie,XGBoostPoisson,LogisticRegression,NegativeBinomialGLM"
    mlt_ct_fu = {"t1_ct", "t1_fu", "t2_ct", "t2_fu", "t3_fu"}
    s = make_full_combos("w3r_mlT", "cfisch", "bigmem", MEM_BY_ABLATION_BIGMEM,
                         28, False, "ml_tabular", MLT_MODELS, only_combos=mlt_ct_fu)
    all_scripts["cfisch"].extend(s)

    # ═══════════════════════════════════════════════════════════════════
    # 2. L40S PARTITION EXPLOITATION (w3l_*) — npin account, iris-snt QOS
    #    Use for GPU-heavy models to reduce GPU queue backlog
    #    L40S: 48GB VRAM, 515G RAM, PreemptMode=REQUEUE (safe with checkpoint)
    # ═══════════════════════════════════════════════════════════════════

    # Foundation models (LagLlama, Moirai, MoiraiLarge, Moirai2) — _ct/_fu
    # These were OOM on batch in Wave 1, now use l40s for GPU overflow
    fmb_ct_fu = {"t1_ct", "t1_fu", "t2_ct", "t2_fu", "t3_fu"}
    s = make_full_combos("w3l_fmB", "npin", "l40s", MEM_BY_ABLATION_L40S,
                         14, True, "foundation", "LagLlama,Moirai,MoiraiLarge,Moirai2",
                         only_combos=fmb_ct_fu, qos="iris-snt")
    all_scripts["npin"].extend(s)

    # Irregular models (BRITS, CSDI) — all non-_co ablations
    # These were pending on gpu queue, move _ce/_ct/_fu to l40s
    irn_non_co = {"t1_ce", "t1_ct", "t1_fu", "t2_ce", "t2_ct", "t2_fu", "t3_ce", "t3_fu"}
    s = make_full_combos("w3l_irN", "npin", "l40s", MEM_BY_ABLATION_L40S,
                         14, True, "irregular", "BRITS,CSDI",
                         only_combos=irn_non_co, qos="iris-snt")
    all_scripts["npin"].extend(s)

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("Phase 9 Wave 3 — SLURM Script Generation Summary")
    print("=" * 70)
    for acct in ("npin", "cfisch"):
        scripts = all_scripts[acct]
        print(f"\n  {acct}: {len(scripts)} scripts")
        for s in sorted(scripts):
            # Show memory allocation
            path = OUT_DIR / f"{s}.sh"
            mem = "?"
            for line in path.read_text().splitlines():
                if "--mem=" in line:
                    mem = line.split("--mem=")[1].strip()
                    break
            print(f"    {s:30s}  ({mem})")
    total = sum(len(v) for v in all_scripts.values())
    print(f"\n  TOTAL: {total} scripts in {OUT_DIR}")

    # Generate submission helpers
    for acct in ("npin", "cfisch"):
        scripts = sorted(all_scripts[acct])
        if not scripts:
            continue
        helper_name = f"submit_wave3_{acct}.sh"
        helper_path = OUT_DIR / helper_name
        lines = ["#!/usr/bin/env bash",
                 f"# Submit all Phase 9 Wave 3 scripts for {acct}",
                 f"cd {OUT_DIR}", ""]
        for s in scripts:
            lines.append(f"sbatch {s}.sh")
        lines.append(f'\necho "Submitted {len(scripts)} jobs for {acct}"')
        helper_path.write_text("\n".join(lines) + "\n")
        helper_path.chmod(helper_path.stat().st_mode | stat.S_IEXEC)
        print(f"\n  Submission helper: {helper_path}")


if __name__ == "__main__":
    main()
