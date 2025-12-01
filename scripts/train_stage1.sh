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
BASE_MODEL="/content/ICL-Router/models/Qwen2.5-7B-Instruct"
EMBED_MODEL="/content/ICL-Router/models/Qwen3-Embedding-8B"

# --- data ---
DATA_DIR="./data"
TRAIN_FILE="$DATA_DIR/question_train.json"
VAL_FILE="$DATA_DIR/question_test.json"
DATASET_JSON="{\"train\": \"$TRAIN_FILE\", \"validation\": \"$VAL_FILE\"}"

# --- optimisation ---
BATCH_SIZE=4
MAX_LENGTH=1024
LR=2e-5
LLM_LR=5e-6
EPOCHS=3

# --- output ---
OUTPUT_DIR="./checkpoints"
CKPT_KEY="icl_stage1_seed42"
CACHE_FILE="Qwen3-8B-embedding_stage1.pt"

#####################################
# 3) Launch DeepSpeed               #
#####################################
deepspeed --num_gpus "$NUM_GPUS" icl_stage1_query_rec_training.py \
  --base_model_name_or_path        "$BASE_MODEL" \
  --embed_model_name_or_path       "$EMBED_MODEL" \
  --dataset_name                   "$DATASET_JSON" \
  --max_length                     "$MAX_LENGTH" \
  --batch_size                     "$BATCH_SIZE" \
  --lr                             "$LR" \
  --llm_lr                         "$LLM_LR" \
  --llm_unfreeze_epoch             1 \
  --num_train_epochs               "$EPOCHS" \
  --projector_type                 nonlinear \
  --output_dir                     "$OUTPUT_DIR" \
  --ckpt_key                       "$CKPT_KEY" \
  --cached_embedding_file          "$CACHE_FILE" 