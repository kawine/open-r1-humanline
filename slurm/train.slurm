#!/bin/bash
#SBATCH --job-name=open_r1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=150G
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=standard_h100
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=2-00:00:00


if [[ "$*" == *"--help"* ]]; then
  echo "Usage: sbatch slurm/train.slurm [options]"
  echo "Options:"
  echo "  --model MODEL            Model name"
  echo "  --task TASK              Task name (e.g. sft, grpo)"
  echo "  --config SUFFIX          Configuration suffix (e.g. demo, v00.00)"
  echo "  --accelerator CONFIG     Accelerator configuration name (e.g. zero3)"
  echo "  --dp N                   Data parallelism for vLLM server (default: 1)"
  echo "  --tp N                   Tensor parallelism for vLLM server (default: 1)"
  echo "  --args \"ARGS\"          Optional arguments to pass to the training script"
  exit 0
fi


# Cleanup at start (on all allocated nodes)
echo "Cleaning up any previous artifacts..."
rm -rf /dev/shm/*torch* /dev/shm/*python* /dev/shm/*$USER* /dev/shm/nccl-* /dev/shm/cuda_*
rm -rf /tmp/*torch* /tmp/torch_* /tmp/vllm* /tmp/hf_cache*
# If using specific rendezvous files
rm -rf /tmp/rdzv*

# vLLM/Ray locks (if using Ray)
rm -rf /tmp/ray*
rm -rf /dev/shm/plasma_*
rm -rf ~/.ray

# Remove System V shared memory segments owned by you
for id in $(ipcs -m | grep $USER | awk '{print $2}'); do
    echo "Removing shared memory segment $id"
    ipcrm -m $id 2>/dev/null
done

# Remove POSIX shared memory
ls /dev/shm/ | grep -E "sem\.|shm\." | while read f; do
    rm -f /dev/shm/$f 2>/dev/null
done

# 3. Semaphores (can block distributed init)
for id in $(ipcs -s | grep $USER | awk '{print $2}'); do
    echo "Removing semaphore $id"
    ipcrm -s $id 2>/dev/null
done

# File-based locks in tmp
find /tmp -user $USER -name "*.lock" -delete 2>/dev/null
find /tmp -user $USER -name "*_lock" -delete 2>/dev/null

export VLLM_CACHE_ROOT=/tmp/vllm_cache_${SLURM_JOBID}_rank_${SLURM_LOCALID}
export TORCH_COMPILE_CACHE_DIR=/tmp/torch_compile_${SLURM_JOBID}_rank_${SLURM_LOCALID}
export VLLM_PORT=$((8000 + SLURM_LOCALID))

export HF_HOME=/tmp/hf_cache_${SLURM_JOBID}

# Or use more aggressive garbage collection
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True

# Wait for cleanup
sleep 5


# Specific configuration optimized for the Hugging Face Compute Cluster
module load cuda/12.4
set -x -e

source ~/.bashrc
source openr1/bin/activate
START_TIME=$(date +%s)
echo "START TIME: $(date)"

# Refresh Weka on h4 cache
echo "Refreshing Weka filesystem..."
find -L /fsx/h4/ -type f | xargs -d '\n' -r -n512 -P64 weka fs tier fetch

# Default values
MODEL=""
TASK=""
CONFIG_SUFFIX=""
ACCELERATOR=""
DP=1
TP=1
OPTIONAL_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --config)
      CONFIG_SUFFIX="$2"
      shift 2
      ;;
    --accelerator)
      ACCELERATOR="$2"
      shift 2
      ;;
    --dp)
      DP="$2"
      shift 2
      ;;
    --tp)
      TP="$2"
      shift 2
      ;;
    --args)
      OPTIONAL_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$MODEL" || -z "$TASK" || -z "$CONFIG_SUFFIX" || -z "$ACCELERATOR" ]]; then
  echo "Error: Missing required arguments"
  echo "Run with --help for usage information"
  exit 1
fi


CONFIG_FILE=recipes/$MODEL/$TASK/config_$CONFIG_SUFFIX.yaml
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE | awk '{print $2}')

# Split the string into individual arguments
IFS=' ' read -ra ARGS <<< "$OPTIONAL_ARGS"
# Loop through the arguments and find the one with "--gradient_accumulation_steps"
for arg in "${ARGS[@]}"; do
    if [[ "$arg" == "--gradient_accumulation_steps="* ]]; then
        # Extract the value after the equals sign
        GRAD_ACC_STEPS="${arg#*=}"
        break  # Exit the loop once we find the desired argument
    fi
done

echo "Gradient accumulation steps: $GRAD_ACC_STEPS"

MODEL=$(grep 'model_name_or_path:' $CONFIG_FILE | awk '{print $2}')
REVISION=$(grep 'model_revision:' $CONFIG_FILE | head -n 1 | awk '{print $2}')

# Distributed configuration
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}  # First node for main process
MASTER_PORT=6000
TRAIN_NODES=("${NODELIST[@]}")

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

export CMD=" \
    src/open_r1/$TASK.py --config $CONFIG_FILE --vllm_mode colocate $OPTIONAL_ARGS
    "

export LAUNCHER="ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/$ACCELERATOR.yaml  \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend=c10d \
    --max_restarts 1 \
    --tee 3 \
    "
# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
NODELIST=$(IFS=,; echo "${TRAIN_NODES[*]}")

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --nodes=$NUM_NODES \
    --ntasks=$NUM_NODES \
    --nodelist=$NODELIST
    "
srun $SRUN_ARGS bash -c "$LAUNCHER $CMD" 2>&1

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"


# Cleanup at exit (runs even if job fails)
cleanup() {
    echo "Cleaning up..."
    rm -rf /dev/shm/*torch* /dev/shm/*$USER*
    rm -rf /tmp/.torch* /tmp/torch_*
}
trap cleanup EXIT
