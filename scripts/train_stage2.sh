#!/usr/bin/env bash
################################################################################
# Stage-2 Router Fine-Tuning Script
#   • Trains the projector-augmented router (CompositeModel) with DeepSpeed
#   • All paths / hparams are collected in Section 2 for easy editing
#   • Extra CLI flags after the GPU list will be forwarded to the Python script
################################################################################

###########################
# 1) Parse GPU selection  #
###########################
GPUS_STRING="${1:-0}"          # first CLI arg (defaults to "0")
shift || true                  # pass any remaining args through
IFS=',' read -r -a GPUS <<< "$GPUS_STRING"
NUM_GPUS="${#GPUS[@]}"

export CUDA_VISIBLE_DEVICES="$GPUS_STRING"
export NCCL_P2P_DISABLE=1       # can help small-batch multi-GPU runs
export NCCL_IB_DISABLE=1

#####################################
# 2) Paths and hyper-parameters     #
#####################################
# --- model checkpoints ---
BASE_MODEL="./checkpoints/icl_stage1_seed42/epoch_2/llm"            # Stage-1 LLM checkpoint
EMBED_MODEL="/path/to/Qwen3-Embedding-8B"            # Frozen embedding model
PROJECTOR_CKPT="./checkpoints/icl_stage1_seed42/epoch_2/projector"   # Pretrained projector (optional)

# --- expert statistics ---
EXPERT_STATS="./data/experts_information_500.json"

# --- data ---
DATA_DIR="./data"
TRAIN_FILE="$DATA_DIR/train_router.json"
VAL_FILE="$DATA_DIR/test_router.json"
DATASET_JSON="{\"train\": \"$TRAIN_FILE\", \"validation\": \"$VAL_FILE\"}"

# --- optimisation ---
BATCH_SIZE=4
MAX_LENGTH=1024
PROJ_LR=1e-5                  # LR for projector params
LLM_LR=2e-6                   # LR for LLM params
EPOCHS=5
PROJECTOR_TYPE="nonlinear"    # {linear|nonlinear}

# --- output ---
OUTPUT_DIR="./checkpoints"
CKPT_KEY="stage2_router_seed42"
CACHE_FILE="Qwen3-Embedding-8B_500_sft.pt"

#####################################
# 3) Launch DeepSpeed               #
#####################################
echo "Launching Stage-2 router training on $NUM_GPUS GPU(s)…"

deepspeed --num_gpus "$NUM_GPUS" icl_stage2_routing_training.py \
  --base_model_name_or_path        "$BASE_MODEL" \
  --embed_model_name_or_path       "$EMBED_MODEL" \
  --projector_path                 "$PROJECTOR_CKPT" \
  --experts_information_file       "$EXPERT_STATS" \
  --dataset_name                   "$DATASET_JSON" \
  --max_length                     "$MAX_LENGTH" \
  --batch_size                     "$BATCH_SIZE" \
  --proj_lr                        "$PROJ_LR" \
  --llm_lr                         "$LLM_LR" \
  --num_train_epochs               "$EPOCHS" \
  --projector_type                 "$PROJECTOR_TYPE" \
  --output_dir                     "$OUTPUT_DIR" \
  --ckpt_key                       "$CKPT_KEY" \
  --cached_embedding_file          "$CACHE_FILE" \
  --seed                           42 