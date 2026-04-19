#!/bin/bash

# 可配置：数据集根目录（请根据实际情况修改）
DATASET_ROOT="./Multivariate_ts"

# 确保出错后继续执行
set +e

echo "Starting experiments with -lim=20 for all datasets in $DATASET_ROOT..."

# 检查目录是否存在
if [ ! -d "$DATASET_ROOT" ]; then
    echo "❌ Error: Dataset root directory '$DATASET_ROOT' does not exist."
    exit 1
fi

# 获取所有一级子文件夹（即数据集名称）
# - 排除隐藏文件夹（以 . 开头）
# - 仅取目录
datasets=()
while IFS= read -r -d '' dir; do
    datasets+=("$(basename "$dir")")
done < <(find "$DATASET_ROOT" -maxdepth 1 -type d ! -name ".*" ! -name "$(basename "$DATASET_ROOT")" -print0)

# 如果没有找到任何数据集
if [ ${#datasets[@]} -eq 0 ]; then
    echo "⚠️ No datasets found in $DATASET_ROOT"
    exit 0
fi

echo "Found datasets: ${datasets[*]}"

# 对每个数据集运行一次训练（20GB）
for ds in "${datasets[@]}"; do
    echo "Running: python UEA.py $ds -lim=20 -de=20GB -b=2"
    python UEA.py "$ds" -lim=18 -de=18GB -b=2 || echo "⚠️ Failed: $ds -lim=20"
done

echo "All experiments completed (with possible failures)."