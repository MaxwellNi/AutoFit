#!/bin/bash
# Block 3 Benchmark - Consolidation & Audit Script
# 在所有 job 完成后运行此脚本
#
# 使用方法：./run_block3_consolidate.sh

set -e

cd ~/projects/repo_root

STAMP="20260203_225620"
RUN_ROOT="runs/benchmarks/block3_${STAMP}_4090_standard"
CONSOLIDATED="${RUN_ROOT}/_consolidated"
AUDIT_FILE="${RUN_ROOT}/_audit.md"

echo "=============================================="
echo "Block 3 Benchmark Consolidation & Audit"
echo "RUN_ROOT: ${RUN_ROOT}"
echo "=============================================="

# 1. 统计完成情况
echo ""
echo "=== 1. 完成度统计 ==="
TOTAL_SHARDS=$(find "${RUN_ROOT}" -name "MANIFEST.json" 2>/dev/null | wc -l)
COMPLETED_SHARDS=$(find "${RUN_ROOT}" -name "metrics.json" 2>/dev/null | wc -l)
echo "Total shards: ${TOTAL_SHARDS}"
echo "Completed (with metrics): ${COMPLETED_SHARDS}"

# 2. 运行 consolidation
echo ""
echo "=== 2. 运行 Consolidation ==="
python scripts/consolidate_block3_results.py \
    --input-dir "${RUN_ROOT}" \
    --output-dir "${CONSOLIDATED}"

# 3. 生成审计报告
echo ""
echo "=== 3. 生成审计报告 ==="

cat > "${AUDIT_FILE}" << EOF
# Block 3 Benchmark Audit Report

## 基本信息
- **生成时间**: $(date -Iseconds)
- **Git Hash**: $(git rev-parse --short HEAD)
- **Python 版本**: $(python --version 2>&1)
- **Freeze STAMP**: ${STAMP}
- **运行目录**: ${RUN_ROOT}

## Freeze Pointer 摘要
\`\`\`yaml
$(head -30 docs/audits/FULL_SCALE_POINTER.yaml)
\`\`\`

## 完成度统计
- **总 Shard 数**: ${TOTAL_SHARDS}
- **成功 Shard 数**: ${COMPLETED_SHARDS}
- **失败 Shard 数**: $((TOTAL_SHARDS - COMPLETED_SHARDS))

## 失败清单
EOF

# 列出失败的 shard
echo "### 缺失 metrics.json 的 Shard" >> "${AUDIT_FILE}"
echo '```' >> "${AUDIT_FILE}"
for manifest in $(find "${RUN_ROOT}" -name "MANIFEST.json"); do
    dir=$(dirname "$manifest")
    if [ ! -f "${dir}/metrics.json" ]; then
        echo "$dir"
        # 打印错误信息
        cat "$manifest" | python -c "import sys,json; m=json.load(sys.stdin); print(f'  Status: {m.get(\"status\")}'); print(f'  Error: {m.get(\"error\", \"N/A\")}')"
    fi
done >> "${AUDIT_FILE}"
echo '```' >> "${AUDIT_FILE}"

# 4. 添加 Leaderboard
echo "" >> "${AUDIT_FILE}"
echo "## Leaderboard" >> "${AUDIT_FILE}"
echo "" >> "${AUDIT_FILE}"

if [ -f "${CONSOLIDATED}/consolidated_metrics.csv" ]; then
    echo '```' >> "${AUDIT_FILE}"
    python -c "
import pandas as pd
df = pd.read_csv('${CONSOLIDATED}/consolidated_metrics.csv')
# 按 task 分组，找最佳模型
for task in df['task'].unique():
    task_df = df[df['task'] == task]
    print(f'=== {task} ===')
    for ablation in task_df['ablation'].unique():
        abl_df = task_df[task_df['ablation'] == ablation]
        if 'mae' in abl_df.columns and not abl_df['mae'].isna().all():
            best = abl_df.loc[abl_df['mae'].idxmin()]
            print(f'{ablation}: {best[\"model_name\"]} (MAE={best[\"mae\"]:.2f}, RMSE={best.get(\"rmse\", \"N/A\")})')
    print()
" >> "${AUDIT_FILE}"
    echo '```' >> "${AUDIT_FILE}"
fi

echo ""
echo "=============================================="
echo "Consolidation 完成！"
echo "审计报告: ${AUDIT_FILE}"
echo "汇总表格: ${CONSOLIDATED}/"
echo "=============================================="

# 打印 leaderboard 到终端
echo ""
cat "${CONSOLIDATED}/summary_stats.json" 2>/dev/null || echo "(No summary stats)"
