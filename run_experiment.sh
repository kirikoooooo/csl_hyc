#!/bin/bash

# 确保出错后继续执行（默认行为，显式声明更清晰）
set +e

echo "Starting experiments..."

# === Cricket 数据集 ===
# python UEA.py Cricket -lim=0.25 -de=0.25GB -b=8 || echo "⚠️ Failed: Cricket -lim=0.25"
# python UEA.py Cricket -lim=0.5 -de=0.5GB -b=8 || echo "⚠️ Failed: Cricket -lim=0.5"
# python UEA.py Cricket -lim=1 -de=1GB -b=8 || echo "⚠️ Failed: Cricket -lim=1"
# python UEA.py Cricket -lim=1.5 -de=1.5GB -b=8 || echo "⚠️ Failed: Cricket -lim=1.5"
# python UEA.py Cricket -lim=2 -de=2GB -b=8 || echo "⚠️ Failed: Cricket -lim=2"
# python UEA.py Cricket -lim=2.5 -de=2.5GB -b=8 || echo "⚠️ Failed: Cricket -lim=2.5"
# python UEA.py Cricket -lim=3 -de=3GB -b=8 || echo "⚠️ Failed: Cricket -lim=3"
# python UEA.py Cricket -lim=4 -de=4GB -b=8 || echo "⚠️ Failed: Cricket -lim=4"

# python UEA.py Cricket -lim=8 -de=8GB -b=64  -logdir=cricket_logs_monet -algo=monet|| echo "⚠️ Failed: Cricket -lim=0.5"
# python UEA.py Cricket -lim=10 -de=10GB -b=64 -logdir=cricket_logs_monet -algo=monet|| echo "⚠️ Failed: Cricket -lim=1"
# python UEA.py Cricket -lim=12 -de=12GB -b=64 -logdir=cricket_logs_monet -algo=monet|| echo "⚠️ Failed: Cricket -lim=1.5"
# python UEA.py Cricket -lim=14 -de=14GB -b=64 -logdir=cricket_logs_monet -algo=monet|| echo "⚠️ Failed: Cricket -lim=2"
# python UEA.py Cricket -lim=16 -de=16GB -b=64 -logdir=cricket_logs_monet -algo=monet|| echo "⚠️ Failed: Cricket -lim=2.5"
# python UEA.py Cricket -lim=18 -de=18GB -b=64 -logdir=cricket_logs_monet -algo=monet|| echo "⚠️ Failed: Cricket -lim=3"

 #python UEA.py Cricket -lim=18 -de=18GB -b=64  -logdir=./ -algo=monet|| echo "⚠️ Failed: Cricket -lim=3"
# 定义算法列表
algos=("monet" "checkmate" "diff" "ga" "mimose")

# 定义 lim 值（de 与 lim 数值相同，单位 GB）
lim_values=(1 2 3 4 5 6)

# 遍历每个算法
for algo in "${algos[@]}"; do
    logdir="cricket_logs_${algo}"
    for lim in "${lim_values[@]}"; do
        de="${lim}GB"
        cmd="python UEA.py PenDigits -lim=${lim} -de=${de} -b=256 -logdir=${logdir} -algo=${algo}"
        fail_msg="⚠️ Failed: Cricket -algo=${algo} -lim=${lim}"
        eval "$cmd" || echo "$fail_msg"
    done
done
# # === PEMS-SF 数据集 ===

# python UEA.py PEMS-SF -lim=1 -de=1GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=1"
# python UEA.py PEMS-SF -lim=2 -de=2GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=2"
# python UEA.py PEMS-SF -lim=3 -de=3GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=3"
# python UEA.py PEMS-SF -lim=4 -de=4GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=4"
# python UEA.py PEMS-SF -lim=5 -de=5GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=5"
# python UEA.py PEMS-SF -lim=6 -de=6GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=6"
# # python UEA.py PEMS-SF -lim=8 -de=8GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=8"
# # python UEA.py PEMS-SF -lim=12 -de=12GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=12"
# # python UEA.py PEMS-SF -lim=16 -de=16GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=16"



# # === DuckDuckGeese 数据集 ===
# # python UEA.py DuckDuckGeese -lim=0.5 -de=0.5GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=0.5"
# # python UEA.py DuckDuckGeese -lim=1 -de=1GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=1"
# # python UEA.py DuckDuckGeese -lim=2 -de=2GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=2"
# # python UEA.py DuckDuckGeese -lim=4 -de=4GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=4"
# # python UEA.py DuckDuckGeese -lim=8 -de=8GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=8"
# python UEA.py DuckDuckGeese -lim=10 -de=10GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=10"
# python UEA.py DuckDuckGeese -lim=12 -de=12GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=12"
# python UEA.py DuckDuckGeese -lim=14 -de=14GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=14"
# python UEA.py DuckDuckGeese -lim=16 -de=16GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=16"
# python UEA.py DuckDuckGeese -lim=18 -de=18GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=18"
# python UEA.py DuckDuckGeese -lim=20 -de=20GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=20"
# python UEA.py DuckDuckGeese -lim=22 -de=22GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=22"
# python UEA.py DuckDuckGeese -lim=24 -de=24GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=24"

# # === MotorImagery 数据集 ===

# # python UEA.py MotorImagery -lim=2 -de=2GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=2"
# # python UEA.py MotorImagery -lim=4 -de=4GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=4"
# # python UEA.py MotorImagery -lim=6 -de=6GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=6"
# # python UEA.py MotorImagery -lim=8 -de=8GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=8"
# python UEA.py MotorImagery -lim=10 -de=10GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=10"
# python UEA.py MotorImagery -lim=12 -de=12GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=12"
# python UEA.py MotorImagery -lim=14 -de=14GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=14"
# python UEA.py MotorImagery -lim=16 -de=16GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=16"
# python UEA.py MotorImagery -lim=18 -de=18GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=18"
# python UEA.py MotorImagery -lim=20 -de=20GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=20"
# python UEA.py MotorImagery -lim=22 -de=22GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=22"
# python UEA.py MotorImagery -lim=24 -de=24GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=24"
# echo "All experiments completed (with possible failures)."