#!/usr/bin/env python3
"""Generate Phase 10 SLURM scripts for new TSLib models.

Batch D (npin): KANAD, FITS, SparseTSF, CATS
Batch E (cfisch): Fredformer, CycleNet, xPatch, FilterTS
"""
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = "/work/projects/eint/logs/phase10"

TASKS = {
    "task1_outcome": ["core_only", "core_edgar", "core_text", "full"],
    "task2_forecast": ["core_only", "core_edgar", "core_text", "full"],
    "task3_risk_adjust": ["core_only", "core_edgar", "full"],
}
ABL_SHORT = {"core_only": "co", "core_edgar": "ce", "core_text": "ct", "full": "fu"}
TASK_SHORT = {"task1_outcome": "t1", "task2_forecast": "t2", "task3_risk_adjust": "t3"}

# Split models across two accounts
BATCH_D = "KANAD,FITS,SparseTSF,CATS"        # npin
BATCH_E = "Fredformer,CycleNet,xPatch,FilterTS"  # cfisch

TEMPLATE_NPIN = r"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
cd /home/users/npin/repo_root
INSIDER_PY="${{CONDA_PREFIX}}/bin/python3"
if [[ ! -x "${{INSIDER_PY}}" ]]; then
  echo "FATAL: insider python missing: ${{INSIDER_PY}}"; exit 2
fi
echo "============================================================"
echo "Phase 10 New Models | Job ${{SLURM_JOB_ID}} on $(hostname)"
echo "$(date -Iseconds) | Python: $(python3 -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

{run_block}
echo "All done: $(date -Iseconds)"
"""

TEMPLATE_CFISCH = r"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
cd /work/projects/eint/repo_root
INSIDER_PY="${{CONDA_PREFIX}}/bin/python3"
if [[ ! -x "${{INSIDER_PY}}" ]]; then
  echo "FATAL: insider python missing: ${{INSIDER_PY}}"; exit 2
fi
echo "============================================================"
echo "Phase 10 New Models | Job ${{SLURM_JOB_ID}} on $(hostname)"
echo "$(date -Iseconds) | Python: $(python3 -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

{run_block}
echo "All done: $(date -Iseconds)"
"""


def gen_run_block(task: str, ablation: str, models: str) -> str:
    return f'''echo ">>> {task} | {ablation} | Models: {models}"
"${{INSIDER_PY}}" scripts/run_block3_benchmark_shard.py \\
    --task {task} --category tslib_sota --ablation {ablation} \\
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/{task}/tslib_sota/{ablation} \\
    --seed 42 --no-verify-first --models {models}
echo "<<< Done {ablation}: $(date -Iseconds)"'''


def main():
    os.makedirs(os.path.join(SCRIPTS_DIR, "npin"), exist_ok=True)
    os.makedirs(os.path.join(SCRIPTS_DIR, "cfisch"), exist_ok=True)

    scripts = {"npin": [], "cfisch": []}

    for task, ablations in TASKS.items():
        ts = TASK_SHORT[task]
        for abl in ablations:
            abl_s = ABL_SHORT[abl]
            mem = "640G" if abl == "full" else "320G"

            # Batch D — npin
            job_name = f"p10_tsD_{ts}_{abl_s}"
            run_block = gen_run_block(task, abl, BATCH_D)
            script = TEMPLATE_NPIN.format(
                job_name=job_name, mem=mem, log_dir=LOG_DIR,
                run_block=run_block
            )
            path = os.path.join(SCRIPTS_DIR, "npin", f"{job_name}.sh")
            with open(path, "w") as f:
                f.write(script)
            scripts["npin"].append(path)

            # Batch E — cfisch
            job_name = f"p10_tsE_{ts}_{abl_s}"
            run_block = gen_run_block(task, abl, BATCH_E)
            script = TEMPLATE_CFISCH.format(
                job_name=job_name, mem=mem, log_dir=LOG_DIR,
                run_block=run_block
            )
            path = os.path.join(SCRIPTS_DIR, "cfisch", f"{job_name}.sh")
            with open(path, "w") as f:
                f.write(script)
            scripts["cfisch"].append(path)

    # Generate submission scripts
    submit_npin = os.path.join(SCRIPTS_DIR, "submit_npin.sh")
    with open(submit_npin, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(f"mkdir -p {LOG_DIR}\n")
        f.write(f"echo 'Submitting {len(scripts['npin'])} Phase 10 npin jobs...'\n")
        for p in sorted(scripts["npin"]):
            f.write(f"sbatch {p}\n")
        f.write("echo 'All npin jobs submitted.'\n")
    os.chmod(submit_npin, 0o755)

    submit_cf = os.path.join(SCRIPTS_DIR, "submit_cfisch.sh")
    with open(submit_cf, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(f"mkdir -p {LOG_DIR}\n")
        f.write(f"echo 'Submitting {len(scripts['cfisch'])} Phase 10 cfisch jobs...'\n")
        for p in sorted(scripts["cfisch"]):
            f.write(f"sbatch {p}\n")
        f.write("echo 'All cfisch jobs submitted.'\n")
    os.chmod(submit_cf, 0o755)

    print(f"Generated {len(scripts['npin'])} npin scripts in {SCRIPTS_DIR}/npin/")
    print(f"Generated {len(scripts['cfisch'])} cfisch scripts in {SCRIPTS_DIR}/cfisch/")
    print(f"Submit with:\n  bash {submit_npin}\n  ssh iris-cf 'bash {submit_cf}'")


if __name__ == "__main__":
    main()
