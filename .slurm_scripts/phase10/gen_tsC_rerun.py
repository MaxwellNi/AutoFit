#!/usr/bin/env python3
"""Generate tsC re-run scripts (ETSformer,LightTS,Pyraformer,Reformer) after tsA/tsB concurrent-write data loss."""
import os

MODELS = "ETSformer,LightTS,Pyraformer,Reformer"
# Dependency: wait for ALL currently running tsA co/ce/ct and tsB co/ce/ct jobs
TSA_JOBS = "5217484:5217485:5217486:5217488:5217489:5217490:5217492:5217493"
TSB_JOBS = "5217550:5217551:5217552:5217554:5217555:5217556:5217558:5217559"
DEPS = f"{TSA_JOBS}:{TSB_JOBS}"

TASKS = {
    "task1_outcome": ["core_only", "core_edgar", "core_text"],
    "task2_forecast": ["core_only", "core_edgar", "core_text"],
    "task3_risk_adjust": ["core_only", "core_edgar"],
}
ABL_SHORT = {"core_only": "co", "core_edgar": "ce", "core_text": "ct"}

TEMPLATE = """#!/usr/bin/env bash
#SBATCH --job-name={jobname}
#SBATCH --account={account}
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=320G
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase10/{jobname}_%j.out
#SBATCH --error=/work/projects/eint/logs/phase10/{jobname}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --dependency=afterany:{deps}

set -e
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
cd {workdir}
INSIDER_PY="${{CONDA_PREFIX}}/bin/python3"
if [[ ! -x "${{INSIDER_PY}}" ]]; then
  echo "FATAL: insider python missing: ${{INSIDER_PY}}"; exit 2
fi
echo "============================================================"
echo "tsC Re-run (data-loss recovery) | Job ${{SLURM_JOB_ID}} on $(hostname)"
echo "$(date -Iseconds) | Python: $(python3 -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

echo "Task: {task} | Abl: {ablation} | Models: {models}"
"${{INSIDER_PY}}" scripts/run_block3_benchmark_shard.py \\
    --task {task} --category tslib_sota --ablation {ablation} \\
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/{task}/tslib_sota/{ablation} --seed 42 \\
    --no-verify-first --models {models}
echo "Done: $(date -Iseconds)"
"""

# Split evenly: npin gets t1 + t2_co, cfisch gets t2_ce + t2_ct + t3
script_count = 0
for task, ablations in TASKS.items():
    tshort = {"task1_outcome":"t1","task2_forecast":"t2","task3_risk_adjust":"t3"}[task]
    for abl in ablations:
        ashort = ABL_SHORT[abl]
        # Distribute: npin takes t1+t3, cfisch takes t2
        if task in ("task1_outcome", "task3_risk_adjust"):
            account = "npin"
            workdir = "/home/users/npin/repo_root"
            acct_dir = "npin"
        else:
            account = "christian.fisch"
            workdir = "/work/projects/eint/repo_root"
            acct_dir = "cfisch"

        jobname = f"p10r_tsC_{tshort}_{ashort}"
        outdir = os.path.join(os.path.dirname(__file__), acct_dir)
        os.makedirs(outdir, exist_ok=True)
        script = TEMPLATE.format(
            jobname=jobname, account=account, workdir=workdir,
            task=task, ablation=abl, models=MODELS, deps=DEPS,
        )
        path = os.path.join(outdir, f"{jobname}.sh")
        with open(path, "w") as f:
            f.write(script)
        os.chmod(path, 0o755)
        print(f"  [{acct_dir}] {path}")
        script_count += 1

print(f"\nGenerated {script_count} tsC re-run scripts with dependency on tsA/tsB completion")
