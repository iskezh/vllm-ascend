#!/bin/bash
# 批量跑 dump 精度对比：custom vs npu(baseline) vs golden(blasst_sim)
# 用法: bash run_dump_batch.sh <tag> [device] [max_n]
#   tag: 输出日志标签（如 baseline_pre_refactor / post_refactor）
#   device: 默认 npu:12
#   max_n: 只跑前 N 个（默认全部）
TAG=${1:-run}
DEV=${2:-npu:12}
MAXN=${3:-99999}
DUMP_DIR=/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2
CUR_DIR=$(cd "$(dirname "$0")" && pwd)
OUT=$CUR_DIR/batch_${TAG}.log
SUM=$CUR_DIR/batch_${TAG}_summary.txt

: > "$OUT"; : > "$SUM"
i=0
# 只跑可回放的 layer 系列（bigdiff/stage 系列为纯输出对比文件，无输入不可回放）
for f in $(ls "$DUMP_DIR"/fia_layer*.pt | head -"$MAXN"); do
    i=$((i+1))
    b=$(basename "$f")
    echo "[$i] $b" >> "$OUT"
    timeout 300 python "$CUR_DIR/run_dump_repro.py" --dump "$f" --device "$DEV" >> "$OUT" 2>&1
    rc=$?
    echo "[$i] $b rc=$rc" >> "$SUM"
done
echo "=== DONE: $i cases, tag=$TAG dev=$DEV ===" >> "$SUM"
