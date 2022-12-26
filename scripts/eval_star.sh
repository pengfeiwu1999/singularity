#!/bin/bash
#SBATCH --partition=XXX  # please specify your partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:1  # number of GPUs
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=sl_ret
#SBATCH --time=48:00:00
#SBATCH --mem=300G

# can add MASTER_PORT to control port for distributed training
# cmd: bash scripts/eval_star.sh  star_key1  star  1  local
exp_name=$1  # note we added ${corpus} prefix automatically
dataset=$2  # star
exp_dir=${SL_EXP_DIR}
ngpus=$3   # number of GPUs to use
mode=$4

output_dir=${exp_dir}/mc_${dataset}/${dataset}_${exp_name}_eval
config_path=./configs/${dataset}.yaml
echo "output dir >> ${output_dir}"

# bash THIS_SCRIPT ... local ...
rdzv_endpoint="${HOSTNAME}:${MASTER_PORT:-40000}"
echo "rdzv_endpoint: ${rdzv_endpoint}"

PYTHONPATH=.:${PYTHONPATH} \
torchrun --nnodes=1 \
--nproc_per_node=${ngpus} \
--rdzv_backend=c10d \
--rdzv_endpoint=${rdzv_endpoint} \
tasks/eval_star.py \
${config_path} \
output_dir=${output_dir} \
wandb.project=sb_ret_${dataset} \
evaluate=True \
${@:5}
