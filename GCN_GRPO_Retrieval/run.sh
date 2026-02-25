#!/bin/bash
# =============================================================================
# GCN-GRPO Evolutionary Retrieval - Universal Run Script
# =============================================================================
# Usage:
#   bash run.sh [--mode MODE] [--data_root PATH] [--model_type TYPE]
#               [--checkpoint PATH] [--save_dir PATH] [options...]
#
# Modes:
#   train          Single training run (default)
#   multi_seed     3-seed training + result aggregation
#   multi_k        Scale study across K values (K=4,8,16,32,64)
#   ablation       Ablation study across reward components
#
# Examples:
#   # Minimal (interactive prompts for required paths):
#   bash run.sh
#
#   # Full explicit:
#   bash run.sh --mode multi_seed \
#               --data_root /path/to/dataset \
#               --model_type retizero \
#               --checkpoint /path/to/model.pth \
#               --save_dir ./checkpoints/exp1
#
#   # EyeCLIP model (no --checkpoint needed):
#   bash run.sh --mode train --model_type eyeclip --data_root /path/to/dataset
# =============================================================================

set -o pipefail

# ── Default hyper-parameters (override via CLI flags) ──────────────────────
MODE="train"
MODEL_TYPE="retizero"
DATA_ROOT=""
CHECKPOINT=""
SAVE_DIR="./checkpoints/grpo_run"
EPOCHS=20
BATCH_SIZE=32
LR="1e-4"
NUM_TRAJ=8
POOL_SIZE=100
INITIAL_K=8
MAX_STEPS=10
SEEDS="1 2 3"
K_VALUES="4 8 16 32 64"
ABLATION_CONFIGS="no_acc no_purity no_density no_step_insert no_step_delete"
DEVICE="cuda"

# ── Parse CLI arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)          MODE="$2";          shift 2 ;;
        --data_root)     DATA_ROOT="$2";     shift 2 ;;
        --model_type)    MODEL_TYPE="$2";    shift 2 ;;
        --checkpoint)    CHECKPOINT="$2";    shift 2 ;;
        --save_dir)      SAVE_DIR="$2";      shift 2 ;;
        --epochs)        EPOCHS="$2";        shift 2 ;;
        --batch_size)    BATCH_SIZE="$2";    shift 2 ;;
        --lr)            LR="$2";            shift 2 ;;
        --num_traj)      NUM_TRAJ="$2";      shift 2 ;;
        --pool_size)     POOL_SIZE="$2";     shift 2 ;;
        --initial_k)     INITIAL_K="$2";     shift 2 ;;
        --max_steps)     MAX_STEPS="$2";     shift 2 ;;
        --seeds)         SEEDS="$2";         shift 2 ;;
        --k_values)      K_VALUES="$2";      shift 2 ;;
        --device)        DEVICE="$2";        shift 2 ;;
        --help|-h)
            head -30 "$0" | grep "^#" | sed 's/^# \{0,2\}//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Interactive prompts for required fields if not provided ─────────────────
if [[ -z "$DATA_ROOT" ]]; then
    read -rp "Enter dataset root path (--data_root): " DATA_ROOT
fi

if [[ "$MODEL_TYPE" == "retizero" && -z "$CHECKPOINT" ]]; then
    read -rp "Enter RetiZero checkpoint path (--checkpoint, or press Enter to skip): " CHECKPOINT
fi

if [[ -z "$DATA_ROOT" ]]; then
    echo "ERROR: --data_root is required."
    exit 1
fi

# ── Build checkpoint flag ───────────────────────────────────────────────────
CKPT_FLAG=""
if [[ -n "$CHECKPOINT" ]]; then
    CKPT_FLAG="--retizero_checkpoint $CHECKPOINT"
fi

# ── Helper: run one training job ────────────────────────────────────────────
run_one() {
    local seed="$1"
    local extra_flags="$2"
    local save_to="$3"
    local run_name="$4"

    mkdir -p "$save_to"
    local log_file="$save_to/training.log"

    echo ">> Seed=${seed}  SaveDir=${save_to}"

    python -u main.py \
        --data_root      "$DATA_ROOT" \
        --model_type     "$MODEL_TYPE" \
        $CKPT_FLAG \
        --epochs         "$EPOCHS" \
        --batch_size     "$BATCH_SIZE" \
        --lr             "$LR" \
        --num_trajectories "$NUM_TRAJ" \
        --retrieval_pool_size "$POOL_SIZE" \
        --initial_k      "$INITIAL_K" \
        --max_steps      "$MAX_STEPS" \
        --device         "$DEVICE" \
        --seed           "$seed" \
        --run_name       "$run_name" \
        --save_dir       "$save_to" \
        $extra_flags \
        2>&1 | tee "$log_file"

    return ${PIPESTATUS[0]}
}

# ── Helper: aggregate JSON results across seeds ─────────────────────────────
aggregate_seeds() {
    local root="$1"
    local seeds_arr=($2)

python - "$root" "${seeds_arr[@]}" <<'PYEOF'
import json, sys, os
import numpy as np

save_dir = sys.argv[1]
seeds    = [int(s) for s in sys.argv[2:]]

hard_m = {k: [] for k in ['ACC','F1','AUC','Sensitivity','Specificity']}
soft_m = {k: [] for k in ['ACC','F1','AUC','Sensitivity','Specificity']}

missing = []
for seed in seeds:
    path = os.path.join(save_dir, f'seed{seed}', f'final_test_results_seed{seed}.json')
    if not os.path.exists(path):
        missing.append(path)
        continue
    with open(path) as f:
        data = json.load(f)
    for k in hard_m:
        hard_m[k].append(data['test_hard_metrics'][k])
        soft_m[k].append(data['test_soft_metrics'][k])

if missing:
    print(f"WARNING: Missing result files: {missing}")

print('\n' + '='*60)
print('AGGREGATED RESULTS')
print('='*60)
for vote, metrics in [('Hard Vote', hard_m), ('Soft Vote', soft_m)]:
    print(f'\n{vote}:')
    for k, vals in metrics.items():
        if vals:
            print(f'  {k:15s}: {np.mean(vals):6.2f} ± {np.std(vals):5.2f}')

out = {
    'seeds': seeds,
    'hard_vote': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'values': v}
                  for k, v in hard_m.items() if v},
    'soft_vote': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'values': v}
                  for k, v in soft_m.items() if v},
}
out_path = os.path.join(save_dir, 'aggregated_results.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=4)
print(f'\n✓ Saved to {out_path}')
print('='*60)
PYEOF
}

# ═══════════════════════════════════════════════════════════════════════════════
# Modes
# ═══════════════════════════════════════════════════════════════════════════════

echo "=================================================="
echo "  GCN-GRPO Retrieval | Mode: ${MODE}"
echo "  Dataset  : ${DATA_ROOT}"
echo "  Model    : ${MODEL_TYPE}"
echo "  Save Dir : ${SAVE_DIR}"
echo "=================================================="

case "$MODE" in

# ── Single training run ─────────────────────────────────────────────────────
train)
    run_one 42 "" "$SAVE_DIR" "single_run"
    if [[ $? -eq 0 ]]; then
        echo "✓ Training complete. Results in: $SAVE_DIR"
    else
        echo "✗ Training failed."; exit 1
    fi
    ;;

# ── Multi-seed (default 3 seeds) ────────────────────────────────────────────
multi_seed)
    seeds_arr=($SEEDS)
    for seed in "${seeds_arr[@]}"; do
        echo "=========================================="
        echo "Seed ${seed}"
        echo "=========================================="
        run_one "$seed" "" "${SAVE_DIR}/seed${seed}" "seed${seed}"
        if [[ $? -ne 0 ]]; then
            echo "✗ Seed ${seed} failed."; exit 1
        fi
        echo "✓ Seed ${seed} done."
    done
    echo "Aggregating results..."
    aggregate_seeds "$SAVE_DIR" "$SEEDS"
    echo "✓ Multi-seed complete."
    ;;

# ── Multi-K scale study ─────────────────────────────────────────────────────
multi_k)
    seeds_arr=($SEEDS)
    k_arr=($K_VALUES)
    for k in "${k_arr[@]}"; do
        echo "=========================================="
        echo "K = ${k}"
        echo "=========================================="
        for seed in "${seeds_arr[@]}"; do
            save_to="${SAVE_DIR}/K${k}/seed${seed}"
            run_one "$seed" "--initial_k $k" "$save_to" "K${k}_seed${seed}"
            if [[ $? -ne 0 ]]; then
                echo "✗ K=${k} Seed=${seed} failed."; exit 1
            fi
            echo "✓ K=${k} Seed=${seed} done."
        done
        aggregate_seeds "${SAVE_DIR}/K${k}" "$SEEDS"
    done
    echo "✓ Multi-K complete."
    ;;

# ── Ablation study ──────────────────────────────────────────────────────────
ablation)
    seeds_arr=($SEEDS)
    cfg_arr=($ABLATION_CONFIGS)
    for cfg in "${cfg_arr[@]}"; do
        echo "=========================================="
        echo "Ablation: --${cfg}"
        echo "=========================================="
        for seed in "${seeds_arr[@]}"; do
            save_to="${SAVE_DIR}/${cfg}/seed${seed}"
            run_one "$seed" "--${cfg}" "$save_to" "${cfg}_seed${seed}"
            if [[ $? -ne 0 ]]; then
                echo "✗ Config=${cfg} Seed=${seed} failed."; exit 1
            fi
            echo "✓ Config=${cfg} Seed=${seed} done."
        done
        aggregate_seeds "${SAVE_DIR}/${cfg}" "$SEEDS"
    done
    echo "✓ Ablation complete."
    ;;

*)
    echo "Unknown mode: $MODE"
    echo "Valid modes: train | multi_seed | multi_k | ablation"
    exit 1
    ;;
esac
