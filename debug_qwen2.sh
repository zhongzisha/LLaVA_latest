#!/bin/bash

#SBATCH --job-name=debug2
#SBATCh --mail-type=ALL
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2,lscratch:400
#SBATCH --time=200:00:00
##SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH --export=ALL



if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th21_ds
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    CODE_ROOT=$FRCE_DATA_ROOT/LLaVA
    MYTMP_DIR=/tmp/zhongz2
    DATA_ROOT=/mnt/gridftp/zhongz2
else
    source /data/zhongz2/anaconda3/bin/activate th21_ds0144
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0
    CODE_ROOT=/home/$USER/LLaVA
    MYTMP_DIR=/lscratch/$SLURM_JOB_ID
    DATA_ROOT=/data/zhongz2/data/LLaVA-Med/video
fi
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH

srun --export ALL --jobid $SLURM_JOB_ID bash debug1.sh "train"

wait


# srun --export ALL --jobid $SLURM_JOB_ID bash debug2.sh   # pretrain
# srun --export ALL --jobid $SLURM_JOB_ID bash debug2_v3.sh  # finetune
srun --export ALL --jobid $SLURM_JOB_ID bash debug4_qwen2.sh  # finetune

wait
exit;


