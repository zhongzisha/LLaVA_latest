#!/bin/bash

# this is a 2 node slurm job example, you will most likely need to adapt --cpus-per-task and --partition

#SBATCH --job-name=debug
#SBATCh --mail-type=ALL
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2,lscratch:300
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

if [ "${SLURM_JOB_NODELIST}" != "" ]; then
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    NNODES=$SLURM_NNODES
    GPUS_PER_NODE=2
else
    MASTER_ADDR=`hostname`
    NNODES=1
    GPUS_PER_NODE=2
fi
MASTER_PORT=25199

if [ ! -e /tmp/$USER ]; then
    mkdir -p /tmp/$USER 
fi
ln -sf $MYTMP_DIR /tmp/$USER/data
json_folder="/tmp/$USER/data/train_json"
image_folder="/tmp/$USER/data"
video_folder="/tmp/$USER/data"

pretrain_data="${json_folder}/llava_image_.json ${json_folder}/llava_med_alignment_500k_cleaned.json"
finetune_data="${json_folder}/llava_med_instruct_60k_cleaned.json ${json_folder}/la_tune_256k.json ${json_folder}/lrv_tune_331k.json ${json_folder}/lvis_tune_220k_.json ${json_folder}/svit_tune_157k.json ${json_folder}/nlp_tune.json"
finetune_data="${json_folder}/llava_med_instruct_60k_cleaned.json ${json_folder}/llava_image_tune_cleaned.json ${json_folder}/nlp_tune.json"


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
    per_device_train_batch_size=1
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
pretrain_output_dir=${DATA_ROOT}/temp_20240405/llava${MY_DEBUG}/${model_name_or_path}/llava-pretrain-${deepspeed_config}-${atten_implementation}-${LORA_POSTFIX}
finetune_output_dir=${pretrain_output_dir}/finetune
moe_output_dir=${finetune_output_dir}/moe
mkdir -p ${moe_output_dir}
num_ckpt_dirs=$(find $finetune_output_dir -maxdepth 1 -type d -name "checkpoint-*" | wc -l)
log_file=$finetune_output_dir/log$((num_ckpt_dirs + 1)).txt
num_train_epochs=1

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    llava/train/train_${atten_implementation}.py \
    ${lora_params} \
    --deepspeed ./scripts/${deepspeed_config}.json \
    --model_name_or_path ${model_name_or_path} \
    --version ${conv_version} \
    --data_path ${finetune_data} \
    --image_folder ${image_folder} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./cache_dir/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature "patch" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type "mlp2x_gelu" \
    --group_by_modality_length True \
    ${data_type_str} \
    --output_dir ${finetune_output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit 1 \
    --learning_rate ${learning_rate}\
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ./cache_dir



































