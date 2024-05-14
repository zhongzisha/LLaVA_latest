#!/bin/bash

#SBATCH --job-name=debug
#SBATCh --mail-type=ALL
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2,lscratch:200
#SBATCH --time=200:00:00
##SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH --export=ALL


set -x -e

echo "START TIME: $(date)"

# function makehostfile() {
# perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
# $slots=2 if $slots==0; # workaround 2 gpu machines
# @nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
# print map { "$b$_ slots=$slots\n" } @nodes'
# }
# makehostfile > hostfile
# cat hostfile

if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th21_ds
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    CODE_ROOT=$FRCE_DATA_ROOT/LLaVA
    MYTMP_DIR=/tmp/zhongz2
    DATA_ROOT=/mnt/gridftp/zhongz2
else
    source /data/zhongz2/anaconda3/bin/activate th21_ds
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0
    CODE_ROOT=/home/$USER/LLaVA
    MYTMP_DIR=/lscratch/$SLURM_JOB_ID
    DATA_ROOT=/data/zhongz2/data/LLaVA-Med/video
fi
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH

export MY_DEBUG=""
LOG_PATH="main_log${MY_DEBUG}.out"

srun --export ALL --jobid $SLURM_JOB_ID bash job1.sh "train"

wait
echo "data done" 

# accelerate estimate-memory

################### stage 1 #######################
do_lora=0
if [ ${do_lora} -eq 1 ]; then
    lora_params="--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5"
else
    lora_params=""
fi

if [ "$CLUSTER_NAME" == "FRCE" ]; then
    per_device_train_batch_size=1
    gradient_accumulation_steps=32
    learning_rate=5e-5
    data_type_str="--bf16 False --fp16 True --tf32 False"
    deepspeed_config=zero2
    atten_implementation=eager    # no flash-attn
else
    # a100
    per_device_train_batch_size=4    # 
    gradient_accumulation_steps=8   # 4 gpus
    learning_rate=1e-3
    data_type_str="--bf16 True --tf32 True"
    deepspeed_config=zero3
    atten_implementation=flash_attention_2  # a100

    # # v100x
    # per_device_train_batch_size=1
    # gradient_accumulation_steps=32
    # learning_rate=1e-3
    # data_type_str="--bf16 False --fp16 True --tf32 False"
    # deepspeed_config=zero2
    # atten_implementation=sdpa
fi
conv_version=plain
model_name_or_path=microsoft/phi-2
# model_name_or_path=BioMistral/BioMistral-7B
model_name_or_path=lmsys/vicuna-7b-v1.5
model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct 
pretrain_output_dir=${DATA_ROOT}/temp_20240514/llava${MY_DEBUG}/${model_name_or_path}/llava-pretrain-${deepspeed_config}-${atten_implementation}-${LORA_POSTFIX}
finetune_output_dir=${pretrain_output_dir}/finetune
moe_output_dir=${finetune_output_dir}/moe
mkdir -p ${moe_output_dir}
NUM_CKPT_DIRS=$(find $pretrain_output_dir -maxdepth 1 -type d -name "checkpoint-*" | wc -l)
LOG_FILE=$pretrain_output_dir/log$((NUM_CKPT_DIRS + 1)).txt

if [ ! -e "${pretrain_output_dir}/mm_projector.bin" ]; then
    srun --export ALL --jobid $SLURM_JOB_ID \
    bash job2_1.sh \
    ${per_device_train_batch_size} \
    ${gradient_accumulation_steps} \
    ${learning_rate} \
    "${data_type_str}" \
    ${deepspeed_config} \
    ${atten_implementation} \
    ${model_name_or_path} \
    "${pretrain_output_dir}" \
    "${lora_params}" \
    ${conv_version} \
    2>&1 | tee -a ${LOG_FILE}.stage1
fi

wait
echo "stage 1 done" 
exit;



################### stage 2 #######################
do_lora=0
if [ ${do_lora} -eq 1 ]; then
    lora_params="--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5"
else
    lora_params=""
fi
if [ "$CLUSTER_NAME" == "FRCE" ]; then
    per_device_train_batch_size=1
    gradient_accumulation_steps=32
    learning_rate=5e-5
    data_type_str="--bf16 False --fp16 True --tf32 False"
    deepspeed_config=zero2
    atten_implementation=eager    # no flash-attn
else
    per_device_train_batch_size=2
    gradient_accumulation_steps=8
    learning_rate=2e-5
    data_type_str="--bf16 True --tf32 True"
    deepspeed_config=zero3
    atten_implementation=flash_attention_2
fi
model_name_or_path=microsoft/phi-2
conv_version=phi
# model_name_or_path=BioMistral/BioMistral-7B
# conv_version=mistral
model_name_or_path=meta-llama/Llama-2-7b-chat-hf
conv_version=llava_llama_2
model_name_or_path=lmsys/vicuna-7b-v1.5
conv_version=v1
model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
conv_version=llava_llama_3
pretrain_output_dir=${DATA_ROOT}/temp_20240512/llava${MY_DEBUG}/${model_name_or_path}/llava-pretrain-${deepspeed_config}-${atten_implementation}-${LORA_POSTFIX}
finetune_output_dir=${pretrain_output_dir}/finetune
moe_output_dir=${finetune_output_dir}/moe
mkdir -p ${moe_output_dir}
NUM_CKPT_DIRS=$(find $finetune_output_dir -maxdepth 1 -type d -name "checkpoint-*" | wc -l)
LOG_FILE=$finetune_output_dir/log$((NUM_CKPT_DIRS + 1)).txt

if [ ! -e "${finetune_output_dir}/config.json" ]; then
    srun --export ALL --jobid $SLURM_JOB_ID \
    bash job2_2.sh \
    ${per_device_train_batch_size} \
    ${gradient_accumulation_steps} \
    ${learning_rate} \
    "${data_type_str}" \
    "zero3" \
    ${atten_implementation} \
    ${model_name_or_path} \
    "${pretrain_output_dir}" \
    "${finetune_output_dir}" \
    "${lora_params}" \
    ${conv_version} \
    2>&1 | tee -a ${LOG_FILE}.stage2
    echo "stage 2 done" 
else
    echo "stage 2 already done"
fi




################### stage 2 anyres #######################
do_lora=0
if [ ${do_lora} -eq 1 ]; then
    lora_params="--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5"
else
    lora_params=""
fi
if [ "$CLUSTER_NAME" == "FRCE" ]; then
    per_device_train_batch_size=1
    gradient_accumulation_steps=32
    learning_rate=5e-5
    data_type_str="--bf16 False --fp16 True --tf32 False"
    deepspeed_config=zero2
    atten_implementation=eager    # no flash-attn
else
    per_device_train_batch_size=4
    gradient_accumulation_steps=16
    learning_rate=2e-5
    data_type_str="--bf16 True --tf32 True"
    deepspeed_config=zero3
    atten_implementation=flash_attention_2
fi
model_name_or_path=microsoft/phi-2
conv_version=phi
# model_name_or_path=BioMistral/BioMistral-7B
# conv_version=mistral
model_name_or_path=meta-llama/Llama-2-7b-chat-hf
conv_version=llava_llama_2
model_name_or_path=lmsys/vicuna-7b-v1.5
conv_version=v1
model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
conv_version=llava_llama_3_v2
pretrain_output_dir=${DATA_ROOT}/temp_20240512/llava${MY_DEBUG}/${model_name_or_path}/llava-pretrain-${deepspeed_config}-${atten_implementation}-${LORA_POSTFIX}
finetune_output_dir=${pretrain_output_dir}/finetune_anyres
moe_output_dir=${finetune_output_dir}/moe
mkdir -p ${moe_output_dir}
NUM_CKPT_DIRS=$(find $finetune_output_dir -maxdepth 1 -type d -name "checkpoint-*" | wc -l)
LOG_FILE=$finetune_output_dir/log$((NUM_CKPT_DIRS + 1)).txt

if [ ! -e "${finetune_output_dir}/config.json" ]; then
    srun --export ALL --jobid $SLURM_JOB_ID \
    bash job2_2_anyres.sh \
    ${per_device_train_batch_size} \
    ${gradient_accumulation_steps} \
    ${learning_rate} \
    "${data_type_str}" \
    "zero3" \
    ${atten_implementation} \
    ${model_name_or_path} \
    "${pretrain_output_dir}" \
    "${finetune_output_dir}" \
    "${lora_params}" \
    ${conv_version} \
    2>&1 | tee -a ${LOG_FILE}.stage2
fi
echo "stage 2 done" 
exit;



















