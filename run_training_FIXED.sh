#!/bin/bash
#SBATCH -J OCC_C_FT-TRAIN
#SBATCH -A LITVINE-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # one launcher per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=32
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --export=ALL,WANDB_API_KEY="1486e4d36eab2571567d82139a82b9364d1f93b7"

# --- env ---
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl
module load cuda/12.1
source /home/adl38/occ_venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# --- paths/args ---
REPO=/rds/user/adl38/hpc-work/OCCPAST2
ENTRY=$REPO/finetune.py
DATA=$REPO/Data/Training_data_other/trainingset2_20.csv
OUT=$REPO/Data/pst2
MODELS=$REPO/Data/models/basemodel.bin
DESCRIPTIONS=$REPO/Data/Training_data_other/pst2_descriptions.csv
SCHEME=$REPO/Data/Training_data_other/pst2.csv

run_suffix="MULTILINGUAL_CURRICULUM_BSINCREASE"
wandb_project="DEBUG_FT"

[[ -f "$ENTRY"  ]] || { echo "Missing $ENTRY";  exit 2; }
[[ -f "$DATA"   ]] || { echo "Missing $DATA";   exit 2; }
[[ -f "$MODELS" ]] || { echo "Missing $MODELS"; exit 2; }

# IMPORTANT: pass *per-GPU* batch for DDP
PER_GPU_BS=512
NUM_WORKERS=8               # per launcher process

# master address/port for torchrun
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(( 10000 + (RANDOM % 50000) ))
export MASTER_PORT
NODE_RANK=$SLURM_PROCID
NNODES=$SLURM_JOB_NUM_NODES
export NCCL_SOCKET_IFNAME=ib0   # If you have InfiniBand

SAVE_DIR="$OUT/mixer-pst2_${run_suffix}"
mkdir -p "$SAVE_DIR"

export WANDB_MODE=online
export WANDB_DIR="$SAVE_DIR"
export WANDB_SILENT=true
export WANDB_RUN_GROUP="pst2_${run_suffix}"
export WANDB_NAME="job${SLURM_JOB_ID}-node${SLURM_NODEID}"
export WANDB_TAGS="slurm,${run_suffix}"
export WANDB_NOTES="Training OCC-C $(date)"

# Non-interactive login (creates ~/.netrc if needed). Won't prompt or block.
wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true

# Use a fresh run dir so old offline settings can't leak in.
cd "$RUN_DIR"

# Flip any per-dir 'offline' flag to online for this CWD.
wandb online >/dev/null 2>&1 || true
# If a settings file already exists and still says offline, force replace:
if [ -f wandb/settings ]; then
  sed -i 's/^mode: *offline$/mode: online/' wandb/settings || true
fi

# --- after your WANDB setup etc. ---

NNODES=$SLURM_JOB_NUM_NODES
NPROC_PER_NODE=4

# Choose a stable rendezvous endpoint
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(( 10000 + (RANDOM % 50000) ))
export MASTER_ADDR MASTER_PORT

workdir="$SLURM_SUBMIT_DIR"
cd "$workdir"
echo -e "Changed directory to $(pwd).\n"

JOBID=$SLURM_JOB_ID
echo -e "JobID: $JOBID\n======"
echo "Time: $(date)"
echo "Running on master node: $(hostname)"
echo "Current directory: $(pwd)"

echo -e "\nExecuting command on each node via srun...\n"

# One launcher per node; each launcher spawns 4 ranks on its node
srun --export=ALL --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  /usr/bin/bash -c '

    torchrun \
      --nnodes '"$NNODES"' \
      --nproc_per_node '"$NPROC_PER_NODE"' \
      --node_rank="$SLURM_PROCID" \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      "'"$ENTRY"'" \
      --save-path "'"$SAVE_DIR"'" \
      --target-cols pst2_1 pst2_2 pst2_3 pst2_4 pst2_5 \
      --all-codes-file "'"$SCHEME"'" \
      --all-codes-col label \
      --seq2seq-weight 0.1 \
      --initial-checkpoint "'"$MODELS"'" \
      --only-encoder \
      --num-epochs 2000 \
      --block-size 8 \
      --input-col occ1 \
      --language-col lang \
      --dataset "'"$DATA"'" \
      --batch-size '"$PER_GPU_BS"' \
      --eval-interval 1000 \
      --save-interval 5000 \
      --use-within-block-sep \
      --drop-bad-labels \
      --num-workers '"$NUM_WORKERS"' \
      --use-amp \
      --prefetch-factor 16 \
      --pin-memory \
      --persistent-workers \
      --log-wandb \
      --wandb-project-name '"$wandb_project"' \
      --debug-pst2-seed 42 \
      --debug-pst2-samples 5 \
      --min-double-ratio 0.9 \
      --min-double-steps 50000 \
      --late-phase-start-step 150000 \
      --late-phase-batch-sizes 1024 1096 2024 \
      --late-phase-batch-steps 200000 250000 \
      --late-phase-lr-mults 0.7 0.7
'
