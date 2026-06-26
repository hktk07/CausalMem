#!/bin/bash

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

# ===== Model / Task =====
CKPT=llava-ov   # llava-onevision-qwen2-7b-ov
CONV_MODE=qwen_1_5
TASK=streamingbench
echo "$TASK"

# ===== GPU setting =====
GPULIST=(0 1 2 3 4 5 6 7)
CHUNKS=${#GPULIST[@]}

# ===== Method =====
METHOD=causal_mem

# ===== FOSS parameters (single setting only) =====
export FOSS_BUDGET=12000
export FOSS_DECAY=0.9
export FOSS_K_MAX=64
export FOSS_MAX_NEW_BASIS=8
export FOSS_TIME_WEIGHT=0.8

# ===== output dir =====
if [ "$METHOD" == "baseline" ]; then
    SAVE_DIR=$(basename "$CKPT")_${CONV_MODE}_frames_${FRAMES}_${QUESTION_TYPE}_${METHOD}
else
    SAVE_DIR=$(basename "$CKPT")_frames_${FRAMES}_${METHOD}_BUDGET_${FOSS_BUDGET}_K-MAX_${FOSS_K_MAX}_DECAY_${FOSS_DECAY}_MAX-NEW-BASIS_${FOSS_MAX_NEW_BASIS}_time_${FOSS_TIME_WEIGHT}
fi

echo "$SAVE_DIR"

# ===== run =====
for IDX in "${!GPULIST[@]}"; do
    GPU_ID=${GPULIST[$IDX]}
    echo "chunk ${IDX} -> GPU ${GPU_ID}"

    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    METHOD=$METHOD \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python3 -W ignore ./eval/modeling_$TASK.py \
        --model-path "$CKPT" \
        --video-dir ./streamingbench/realtime_video \
        --gt-file ./streamingbench/StreamingBench/Real_Time_Visual_Understanding.csv \
        --output-dir ./outputs/$TASK/$SAVE_DIR \
        --output-name pred \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode $CONV_MODE &
done