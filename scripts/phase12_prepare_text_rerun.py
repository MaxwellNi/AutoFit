#!/usr/bin/env python3
"""Phase 12: Generate SLURM scripts for core_text/full re-run after text embeddings.

After text_embeddings.parquet is generated, the existing core_text and full
ablation results are invalid (they had no actual text features — raw text columns
were silently dropped by select_dtypes). This script:

1. Validates text_embeddings.parquet exists and is complete
2. Backs up old core_text/full metrics.json files (rename, not delete)
3. Generates SLURM scripts for all (task, category, ablation) re-runs
4. Creates a submission helper to submit all via npin + cfisch

Usage:
    python3 scripts/phase12_prepare_text_rerun.py --check-only    # verify readiness
    python3 scripts/phase12_prepare_text_rerun.py --generate      # generate scripts
    python3 scripts/phase12_prepare_text_rerun.py --invalidate    # backup old results
    python3 scripts/phase12_prepare_text_rerun.py --submit        # submit all jobs
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RESULTS_DIR = "runs/benchmarks/block3_phase9_fair"
EMBEDDINGS_PATH = "runs/text_embeddings/text_embeddings.parquet"
SCRIPT_DIR = ".slurm_scripts/phase12/rerun"
LOG_DIR = "/work/projects/eint/logs/phase12"

# GPU categories (need --gres=gpu:1)
GPU_CATEGORIES = {
    "deep_classical": "DeepAR,DilatedRNN,GRU,LSTM,MLP,NBEATS,NHITS,TCN,TFT",
    "transformer_sota": "Autoformer,BiTCN,DLinear,DeepNPTS,FEDformer,Informer,KAN,NBEATSx,NLinear,PatchTST,RMoK,SOFTS,StemGNN,TSMixer,TSMixerx,TiDE,TimeLLM,TimeMixer,TimeXer,TimesNet,VanillaTransformer,iTransformer,xLSTM",
    "foundation": "Chronos,Chronos2,ChronosBolt,LagLlama,MOMENT,Moirai,Moirai2,MoiraiLarge,Sundial,TTM,TimeMoE,Timer,TimesFM,TimesFM2",
    "tslib_sota": "CATS,Crossformer,ETSformer,FITS,KANAD,LightTS,MSGNet,MambaSimple,MultiPatchFormer,PAttn,Pyraformer,Reformer,TimeFilter,WPMixer",
    "irregular": "BRITS,CSDI,GRU-D,SAITS",
}

# CPU categories (bigmem, no GPU)
CPU_CATEGORIES = {
    "ml_tabular": "CatBoost,ExtraTrees,HistGradientBoosting,LightGBM,LightGBMTweedie,MeanPredictor,NegativeBinomialGLM,RandomForest,SeasonalNaive,XGBoost,XGBoostPoisson",
    "statistical": "AutoARIMA,AutoCES,AutoETS,AutoTheta,CrostonClassic,CrostonOptimized,CrostonSBA,DynamicOptimizedTheta,HistoricAverage,Holt,HoltWinters,MSTL,Naive,SF_SeasonalNaive,WindowAverage",
}

# AutoFit handled separately: V734-V736 oracle-leaked (skip), V739 needs GPU
AUTOFIT_V739 = "AutoFitV739"

# Ablation-task matrix: core_text for task1/task2, full for all 3 tasks
RERUN_COMBOS = [
    ("task1_outcome", "core_text"),
    ("task1_outcome", "full"),
    ("task2_forecast", "core_text"),
    ("task2_forecast", "full"),
    ("task3_risk_adjust", "full"),  # no core_text for task3
]

# Memory: with PCA embeddings (64 floats), core_text/full ≈ core_only + ~5GB
# Much less than raw text (200GB+). Conservative allocations:
MEM_GPU = "320G"       # GPU cats: core + embeddings + model
MEM_BIGMEM = "320G"    # CPU cats: same data, no GPU
MEM_V739_GPU = "320G"  # V739 autofit on GPU

NPIN_HEADER = """#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account=npin
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --time=2-00:00:00
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
{gpu_line}#SBATCH --output={log_dir}/{job_name}_%j.out
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
"""

CFISCH_HEADER = """#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account=christian.fisch
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --time=2-00:00:00
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
{gpu_line}#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${{LD_LIBRARY_PATH:-}}"
export HF_HOME="/home/users/npin/.cache/huggingface"
cd /work/projects/eint/repo_root
"""

BODY = """
if [[ ! -x "${{INSIDER_PY}}" ]]; then
  echo "FATAL: insider python missing"; exit 2
fi
echo "============================================================"
echo "Phase 12 Text Re-run | Job ${{SLURM_JOB_ID}} on $(hostname)"
echo "$(date -Iseconds) | Python: $(${{INSIDER_PY}} -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

# Verify text embeddings exist
if [[ ! -f "runs/text_embeddings/text_embeddings.parquet" ]]; then
  echo "FATAL: text_embeddings.parquet not found — cannot run text ablation!"; exit 3
fi
echo "Text embeddings verified: $(ls -lh runs/text_embeddings/text_embeddings.parquet)"

echo "Task: {task} | Cat: {category} | Abl: {ablation} | Models: {models}"
"${{INSIDER_PY}}" scripts/run_block3_benchmark_shard.py \\
    --task {task} --category {category} --ablation {ablation} \\
    --preset full --output-dir {output_dir} --seed 42 \\
    --no-verify-first --models {models}
echo "Done: $(date -Iseconds)"
"""

ABL_SHORT = {"core_text": "ct", "full": "fu"}
TASK_SHORT = {"task1_outcome": "t1", "task2_forecast": "t2", "task3_risk_adjust": "t3"}


def check_embeddings():
    """Check if text_embeddings.parquet exists and is valid."""
    emb_path = Path(EMBEDDINGS_PATH)
    if not emb_path.exists():
        print(f"❌ {EMBEDDINGS_PATH} does NOT exist yet")
        print("   Text embedding generation jobs must complete first.")
        return False
    
    size = emb_path.stat().st_size
    print(f"✅ {EMBEDDINGS_PATH} exists ({size / 1e6:.1f} MB)")
    
    # Quick validation
    try:
        import pandas as pd
        df = pd.read_parquet(emb_path, columns=["entity_id"])
        print(f"   Rows: {len(df):,}")
        if len(df) < 100_000:
            print(f"   ⚠️ Very few rows — may be incomplete")
            return False
    except Exception as e:
        print(f"   ❌ Cannot read parquet: {e}")
        return False
    
    return True


def count_invalid_results():
    """Count core_text/full metrics.json files that need invalidation."""
    count = 0
    records = 0
    for task, abl in RERUN_COMBOS:
        for cat_dir in Path(RESULTS_DIR).joinpath(task).iterdir():
            if not cat_dir.is_dir():
                continue
            mf = cat_dir / abl / "metrics.json"
            if mf.exists():
                count += 1
                with open(mf) as f:
                    data = json.load(f)
                records += len(data)
    return count, records


def invalidate_old_results():
    """Backup old core_text/full metrics.json by renaming."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backed = 0
    for task, abl in RERUN_COMBOS:
        for cat_dir in Path(RESULTS_DIR).joinpath(task).iterdir():
            if not cat_dir.is_dir():
                continue
            mf = cat_dir / abl / "metrics.json"
            if mf.exists():
                backup = mf.with_name(f"metrics_no_text_backup_{ts}.json")
                shutil.move(str(mf), str(backup))
                print(f"  Backed up: {mf} → {backup.name}")
                backed += 1
            # Also backup predictions if present
            pf = cat_dir / abl / "predictions.parquet"
            if pf.exists():
                backup = pf.with_name(f"predictions_no_text_backup_{ts}.parquet")
                shutil.move(str(pf), str(backup))
    print(f"\nBacked up {backed} metrics.json files (renamed, not deleted)")


def generate_scripts():
    """Generate all SLURM scripts for core_text/full re-runs."""
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    scripts_npin = []
    scripts_cfisch = []
    
    # Alternate between npin and cfisch for even distribution
    job_counter = 0
    
    for task, abl in RERUN_COMBOS:
        t_short = TASK_SHORT[task]
        a_short = ABL_SHORT[abl]
        
        # GPU categories
        for cat, models in GPU_CATEGORIES.items():
            job_counter += 1
            is_cfisch = (job_counter % 2 == 0)
            prefix = "cf_" if is_cfisch else ""
            job_name = f"{prefix}p12_{cat[:4]}_{t_short}_{a_short}"
            
            header = CFISCH_HEADER if is_cfisch else NPIN_HEADER
            script = header.format(
                job_name=job_name,
                partition="gpu",
                qos="normal",
                mem=MEM_GPU,
                cpus=14,
                gpu_line="#SBATCH --gres=gpu:1\n",
                log_dir=LOG_DIR,
            ) + BODY.format(
                task=task,
                category=cat,
                ablation=abl,
                models=models,
                output_dir=f"{RESULTS_DIR}/{task}/{cat}/{abl}",
            )
            
            script_path = f"{SCRIPT_DIR}/{job_name}.sh"
            with open(script_path, "w") as f:
                f.write(script)
            os.chmod(script_path, 0o755)
            
            if is_cfisch:
                scripts_cfisch.append(script_path)
            else:
                scripts_npin.append(script_path)
        
        # CPU categories (bigmem)
        for cat, models in CPU_CATEGORIES.items():
            job_counter += 1
            is_cfisch = (job_counter % 2 == 0)
            prefix = "cf_" if is_cfisch else ""
            job_name = f"{prefix}p12_{cat[:4]}_{t_short}_{a_short}"
            
            header = CFISCH_HEADER if is_cfisch else NPIN_HEADER
            script = header.format(
                job_name=job_name,
                partition="bigmem",
                qos="normal",
                mem=MEM_BIGMEM,
                cpus=28,
                gpu_line="",
                log_dir=LOG_DIR,
            ) + BODY.format(
                task=task,
                category=cat,
                ablation=abl,
                models=models,
                output_dir=f"{RESULTS_DIR}/{task}/{cat}/{abl}",
            )
            
            script_path = f"{SCRIPT_DIR}/{job_name}.sh"
            with open(script_path, "w") as f:
                f.write(script)
            os.chmod(script_path, 0o755)
            
            if is_cfisch:
                scripts_cfisch.append(script_path)
            else:
                scripts_npin.append(script_path)
        
        # AutoFit V739 (GPU, only if meaningful — skip oracle-leaked V734-V736)
        job_counter += 1
        is_cfisch = (job_counter % 2 == 0)
        prefix = "cf_" if is_cfisch else ""
        job_name = f"{prefix}p12_af39_{t_short}_{a_short}"
        
        header = CFISCH_HEADER if is_cfisch else NPIN_HEADER
        script = header.format(
            job_name=job_name,
            partition="gpu",
            qos="normal",
            mem=MEM_V739_GPU,
            cpus=14,
            gpu_line="#SBATCH --gres=gpu:1\n",
            log_dir=LOG_DIR,
        ) + BODY.format(
            task=task,
            category="autofit",
            ablation=abl,
            models=AUTOFIT_V739,
            output_dir=f"{RESULTS_DIR}/{task}/autofit/{abl}",
        )
        
        script_path = f"{SCRIPT_DIR}/{job_name}.sh"
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)
        
        if is_cfisch:
            scripts_cfisch.append(script_path)
        else:
            scripts_npin.append(script_path)
    
    print(f"Generated {len(scripts_npin)} npin scripts + {len(scripts_cfisch)} cfisch scripts")
    print(f"  Total: {len(scripts_npin) + len(scripts_cfisch)} scripts in {SCRIPT_DIR}/")
    
    # Generate submission helper
    submit_path = f"{SCRIPT_DIR}/submit_all_phase12_rerun.sh"
    with open(submit_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Phase 12: Submit ALL core_text/full re-run jobs\n")
        f.write("# Run AFTER text_embeddings.parquet is generated and verified\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        f.write("set -e\n")
        f.write("cd /work/projects/eint/repo_root\n\n")
        f.write("# Verify text embeddings exist\n")
        f.write("if [[ ! -f runs/text_embeddings/text_embeddings.parquet ]]; then\n")
        f.write('  echo "FATAL: text_embeddings.parquet not found!"; exit 1\n')
        f.write("fi\n\n")
        f.write(f'echo "Submitting {len(scripts_npin)} npin + {len(scripts_cfisch)} cfisch jobs..."\n\n')
        f.write("# npin jobs (direct sbatch)\n")
        for s in scripts_npin:
            f.write(f"sbatch {s} && echo \"  ✓ {os.path.basename(s)}\"\n")
        f.write("\n# cfisch jobs (via ssh iris-cf)\n")
        for s in scripts_cfisch:
            f.write(f'ssh iris-cf "cd /work/projects/eint/repo_root && sbatch {s}" && echo "  ✓ {os.path.basename(s)}"\n')
        f.write(f'\necho "All {len(scripts_npin) + len(scripts_cfisch)} jobs submitted!"\n')
    os.chmod(submit_path, 0o755)
    print(f"Submission helper: {submit_path}")
    
    return scripts_npin, scripts_cfisch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="Only verify readiness")
    parser.add_argument("--generate", action="store_true", help="Generate SLURM scripts")
    parser.add_argument("--invalidate", action="store_true", help="Backup old core_text/full results")
    parser.add_argument("--submit", action="store_true", help="Submit all jobs")
    args = parser.parse_args()
    
    if args.check_only or not any([args.generate, args.invalidate, args.submit]):
        print("=== Phase 12 Text Re-run Readiness Check ===\n")
        emb_ok = check_embeddings()
        n_files, n_records = count_invalid_results()
        print(f"\nInvalid results to replace: {n_files} files, {n_records} records")
        print(f"  → core_text results = copies of core_only (no text features)")
        print(f"  → full results = copies of core_edgar (no text features)")
        if not emb_ok:
            print("\n⏳ Waiting for text embedding generation to complete.")
            print("   Check: squeue -j 5229600,5229601,5229602,5229603")
        return
    
    if args.generate:
        print("=== Generating Phase 12 Re-run Scripts ===\n")
        generate_scripts()
        return
    
    if args.invalidate:
        print("=== Invalidating Old core_text/full Results ===\n")
        if not check_embeddings():
            print("\n❌ Cannot invalidate until embeddings are ready!")
            sys.exit(1)
        invalidate_old_results()
        return
    
    if args.submit:
        if not check_embeddings():
            print("❌ Cannot submit until embeddings are ready!")
            sys.exit(1)
        
        # Check if old results are invalidated
        n_files, _ = count_invalid_results()
        if n_files > 0:
            print(f"⚠️ {n_files} old core_text/full results still exist.")
            print("   Run with --invalidate first to backup old results.")
            sys.exit(1)
        
        submit_path = f"{SCRIPT_DIR}/submit_all_phase12_rerun.sh"
        if not os.path.exists(submit_path):
            print("❌ Scripts not generated yet! Run with --generate first.")
            sys.exit(1)
        
        print("=== Submitting Phase 12 Re-run Jobs ===")
        subprocess.run(["bash", submit_path], check=True)


if __name__ == "__main__":
    main()
