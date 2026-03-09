#!/usr/bin/env python3
"""Generate Phase 9 Wave 2 SLURM scripts.

Covers:
  1. OOM resubmits  (p9r_*) — fixed memory / partition
  2. New models      (p9n_*) — models without any Phase 9 script
  3. Gap-fill        (p9g_*) — models at 99-103/104 needing a few conditions

Two account templates: npin  (micromamba activate)
                        cfisch (direct insider python)
"""
from pathlib import Path
import stat, textwrap

OUT_DIR = Path(__file__).resolve().parent.parent / ".slurm_scripts" / "phase9"
LOG_BASE = "/work/projects/eint/logs/phase9"

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
    partition: str,        # batch | gpu | bigmem
    mem: str,              # e.g. "256G"
    cpus: int,
    gpu: bool,
    task: str,             # e.g. "task1_outcome"
    ablation: str,         # e.g. "core_only"
    category: str,         # e.g. "autofit"
    models: str,           # comma-separated model names
    preset: str = "full",
    output_dir_override: str | None = None,
) -> str:
    acct = "christian.fisch" if account == "cfisch" else "npin"
    env_block = _cfisch_env_block() if account == "cfisch" else _npin_env_block()
    gres_line = "#SBATCH --gres=gpu:1\n" if gpu else ""
    odir = output_dir_override or f"runs/benchmarks/block3_phase9_fair/{task}/{category}/{ablation}"

    script = f"""#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --account={acct}
#SBATCH --partition={partition}
#SBATCH --qos=normal
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
    # Remove consecutive blank lines (keep single blanks for readability)
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


def make_full_combos(prefix, account, partition, mem, cpus, gpu, category, models,
                     only_combos=None):
    """Generate scripts for all task/ablation combos (or a subset)."""
    scripts = []
    for task, ablations in TASK_ABLATIONS.items():
        for abl in ablations:
            ts = TASK_SHORT[task]
            als = ABL_SHORT[abl]
            combo = f"{ts}_{als}"
            if only_combos and combo not in only_combos:
                continue
            name = f"{prefix}_{combo}"
            content = gen_script(
                name=name, account=account, partition=partition,
                mem=mem, cpus=cpus, gpu=gpu,
                task=task, ablation=abl, category=category, models=models,
            )
            write_script(name, content)
            scripts.append(name)
    return scripts


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_scripts = {"npin": [], "cfisch": []}

    # ═══════════════════════════════════════════════════════════════════
    # 1. OOM RESUBMITS  (p9r_*)  →  cfisch account
    # ═══════════════════════════════════════════════════════════════════

    # af9: 8 OOM (everything except _co which is still running on npin)
    af9_oom = {"t1_ce", "t1_ct", "t1_fu", "t2_ce", "t2_ct", "t2_fu", "t3_ce", "t3_fu"}
    s = make_full_combos("p9r_af9", "cfisch", "bigmem", "256G", 28, False,
                         "autofit", "AutoFitV736", only_combos=af9_oom)
    all_scripts["cfisch"].extend(s)

    # mlT: 10 OOM (everything except t3_co which is running)
    mlt_oom = {"t1_co", "t1_ce", "t1_ct", "t1_fu", "t2_co", "t2_ce", "t2_ct", "t2_fu",
               "t3_ce", "t3_fu"}
    MLT_MODELS = "LightGBM,XGBoost,CatBoost,RandomForest,ExtraTrees,HistGradientBoosting,Ridge,Lasso,ElasticNet,SVR,KNN,QuantileRegressor,MeanPredictor,SeasonalNaive,LightGBMTweedie,XGBoostPoisson,LogisticRegression,NegativeBinomialGLM"
    s = make_full_combos("p9r_mlT", "cfisch", "bigmem", "256G", 28, False,
                         "ml_tabular", MLT_MODELS, only_combos=mlt_oom)
    all_scripts["cfisch"].extend(s)

    # irN: 8 OOM (everything except _co)
    irn_oom = {"t1_ce", "t1_ct", "t1_fu", "t2_ce", "t2_ct", "t2_fu", "t3_ce", "t3_fu"}
    s = make_full_combos("p9r_irN", "cfisch", "gpu", "320G", 14, True,
                         "irregular", "BRITS,CSDI", only_combos=irn_oom)
    all_scripts["cfisch"].extend(s)

    # fmB: 5 OOM — remove Moirai2Large (not in registry)
    fmb_oom = {"t1_ct", "t1_fu", "t2_ct", "t2_fu", "t3_fu"}
    s = make_full_combos("p9r_fmB", "cfisch", "gpu", "500G", 14, True,
                         "foundation", "LagLlama,Moirai,MoiraiLarge,Moirai2",
                         only_combos=fmb_oom)
    all_scripts["cfisch"].extend(s)

    # ═══════════════════════════════════════════════════════════════════
    # 2. NEW MODELS  (p9n_*)
    # ═══════════════════════════════════════════════════════════════════

    # Statistical: 10 new models → cfisch / bigmem
    STAT_NEW = "AutoCES,CrostonClassic,CrostonOptimized,CrostonSBA,DynamicOptimizedTheta,HistoricAverage,Holt,HoltWinters,Naive,WindowAverage"
    s = make_full_combos("p9n_sta", "cfisch", "bigmem", "128G", 28, False,
                         "statistical", STAT_NEW)
    all_scripts["cfisch"].extend(s)

    # Foundation: 3 new models → npin / gpu
    s = make_full_combos("p9n_fmC", "npin", "gpu", "500G", 14, True,
                         "foundation", "Chronos2,TTM,TimerXL")
    all_scripts["npin"].extend(s)

    # TabPFN: 2 new models → npin / gpu
    s = make_full_combos("p9n_tpf", "npin", "gpu", "128G", 14, True,
                         "ml_tabular", "TabPFNClassifier,TabPFNRegressor")
    all_scripts["npin"].extend(s)

    # TSLib (6 missing): → npin / gpu
    TSLIB_NEW = "ETSformer,LightTS,Mamba,Pyraformer,Reformer,TiRex"
    s = make_full_combos("p9n_tsC", "npin", "gpu", "320G", 14, True,
                         "tslib_sota", TSLIB_NEW)
    all_scripts["npin"].extend(s)

    # ═══════════════════════════════════════════════════════════════════
    # 3. GAP-FILL  (p9g_*)  →  npin
    # ═══════════════════════════════════════════════════════════════════

    # 10 transformer_sota at 99/104 — missing task1_outcome core_only + core_text for is_funded
    TRA_GAP_MODELS = "Autoformer,FEDformer,Informer,NBEATSx,PatchTST,TSMixer,TiDE,TimesNet,VanillaTransformer,iTransformer"

    # task1_outcome / core_only — fills 4 missing is_funded horizons (resume skips done)
    content = gen_script(
        "p9g_trCO", "npin", "gpu", "320G", 14, True,
        "task1_outcome", "core_only", "transformer_sota", TRA_GAP_MODELS,
    )
    write_script("p9g_trCO", content)
    all_scripts["npin"].append("p9g_trCO")

    # task1_outcome / core_text — fills 1 missing is_funded h30 (resume skips done)
    content = gen_script(
        "p9g_trCT", "npin", "gpu", "320G", 14, True,
        "task1_outcome", "core_text", "transformer_sota", TRA_GAP_MODELS,
    )
    write_script("p9g_trCT", content)
    all_scripts["npin"].append("p9g_trCT")

    # Sundial + TimesFM2 at 103/104 — missing task2_forecast core_text investors_count h30
    content = gen_script(
        "p9g_fmCT", "npin", "gpu", "500G", 14, True,
        "task2_forecast", "core_text", "foundation", "Sundial,TimesFM2",
    )
    write_script("p9g_fmCT", content)
    all_scripts["npin"].append("p9g_fmCT")

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("Phase 9 Wave 2 — SLURM Script Generation Summary")
    print("=" * 70)
    for acct in ("npin", "cfisch"):
        scripts = all_scripts[acct]
        print(f"\n  {acct}: {len(scripts)} scripts")
        for s in sorted(scripts):
            print(f"    {s}")
    total = sum(len(v) for v in all_scripts.values())
    print(f"\n  TOTAL: {total} scripts in {OUT_DIR}")

    # Generate submission helpers
    for acct in ("npin", "cfisch"):
        scripts = sorted(all_scripts[acct])
        if not scripts:
            continue
        helper_name = f"submit_wave2_{acct}.sh"
        helper_path = OUT_DIR / helper_name
        lines = ["#!/usr/bin/env bash", f"# Submit all Phase 9 Wave 2 scripts for {acct}",
                 f"cd {OUT_DIR}", ""]
        for s in scripts:
            lines.append(f"sbatch {s}.sh")
        lines.append(f'\necho "Submitted {len(scripts)} jobs for {acct}"')
        helper_path.write_text("\n".join(lines) + "\n")
        helper_path.chmod(helper_path.stat().st_mode | stat.S_IEXEC)
        print(f"\n  Submission helper: {helper_path}")


if __name__ == "__main__":
    main()
