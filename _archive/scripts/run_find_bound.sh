#!/bin/bash

algo="checkmate"
logdir="findbound_cricket_logs_${algo}"

low=1100
high=1300
last_success=-1   # 记录最后一个成功的 len

echo "🔍 Searching for maximum feasible len in [$low, $high]..."

while [ $low -le $high ]; do
    mid=$(( (low + high) / 2 ))
    de="${mid}len"
    cmd="python UEA.py Cricket -lim=18 -de=${algo}_bs64_${de} -b=64 -len=${mid} -logdir=${logdir} -algo=${algo}"

    echo "➡️  Testing len=${mid} ..."

    if eval "$cmd"; then
        # ✅ 成功（完整跑完，exit 0）→ 可尝试更长
        echo "✅ Success at len=${mid}"
        last_success=$mid
        low=$((mid + 1))   # ← 往右找更大的可行值
    else
        # ❌ 失败（提前 exit(1)）→ 太长了，缩短
        echo "❌ Failed at len=${mid} (exit code: $?)"
        high=$((mid - 1))  # ← 往左缩
    fi
done

if [ $last_success -eq -1 ]; then
    echo "❗ No successful len found in [$((1100)), $((1200))]"
    exit 1
else
    echo "🎯 Maximum feasible len = $last_success"

    # 【可选】验证：last_success 成功，last_success+1 失败（若在范围内）
    next=$((last_success + 1))
    if [ $next -le 1200 ]; then
        de_next="${next}len"
        cmd_next="python UEA.py Cricket -lim=18 -de=${algo}_bs64_${de_next} -b=64 -len=${next} -logdir=${logdir} -algo=${algo}"
        echo "🛡️ Verifying: len=${next} should fail..."
        if eval "$cmd_next" 2>/dev/null; then
            echo "⚠️  Unexpected: len=${next} also succeeded!"
        else
            echo "✅ Verified: len=${last_success} ✅, len=${next} ❌ → boundary confirmed."
        fi
    else
        echo "✅ len=${last_success} is upper bound (1200)."
    fi
fi