#!/usr/bin/env python3
"""Generate SLURM recovery scripts for WPMixer + Koopa (0-record models)."""
import os

TASKS = {
    "task1_outcome": ["core_only", "core_edgar", "core_text", "full"],
    "task2_forecast": ["core_only", "core_edgar", "core_text", "full"],
    "task3_risk_adjust": ["core_only", "core_edgar", "full"],
}
ABL_SHORT = {"core_only": "co", "core_edgar": "ce", "core_text": "ct", "full": "fu"}

TEMPLATE = """#!/usr/bin/env bash
#SBATCH --job-name={jobname}
#SBATCH --account={account}
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase10/{jobname}_%j.out
#SBATCH --error=/work/projects/eint/logs/phase10/{jobname}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

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
echo "Phase 10 Recovery | Job ${{SLURM_JOB_ID}} on $(hostname)"
echo "$(date -Iseconds) | Python: $(python3 -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

echo "Task: {task} | Cat: tslib_sota | Abl: {ablation} | Models: {models}"
"${{INSIDER_PY}}" scripts/run_block3_benchmark_shard.py \\
    --task {task} --category tslib_sota --ablation {ablation} \\
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/{task}/tslib_sota/{ablation} --seed 42 \\
    --no-verify-first --models {models}
echo "Done: $(date -Iseconds)"
"""

# WPMixer on npin, Koopa on cfisch (evenly distributed)
batches = [
    ("npin", "WPMixer", "npin", "/home/users/npin/repo_root", "p10r_WP"),
    ("cfisch", "Koopa", "christian.fisch", "/work/projects/eint/repo_root", "p10r_KP"),
]

for acct_label, models, account, workdir, prefix in batches:
    outdir = os.path.join(os.path.dirname(__file__), acct_label)
    os.makedirs(outdir, exist_ok=True)
    for task, ablations in TASKS.items():
        tshort = task.split("_")[0][0] + task.split("_")[0][-1]  # t1/t2/t3
        if task == "task1_outcome": tshort = "t1"
        elif task == "task2_forecast": tshort = "t2"
        else: tshort = "t3"
        for abl in ablations:
            ashort = ABL_SHORT[abl]
            mem = "640G" if abl == "full" else "320G"
            jobname = f"{prefix}_{tshort}_{ashort}"
            script = TEMPLATE.format(
                jobname=jobname, account=account, mem=mem,
                workdir=workdir, task=task, ablation=abl, models=models,
            )
            path = os.path.join(outdir, f"{jobname}.sh")
            with open(path, "w") as f:
                f.write(script)
            os.chmod(path, 0o755)
            print(f"  {path}")

print(f"\nGenerated recovery scripts for WPMixer (npin) + Koopa (cfisch)")
