#!/bin/bash

#SBATCH --job-name=debug
#SBATCh --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1,lscratch:200
#SBATCH --time=100:00:00
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

# cp ${DATA_ROOT}/MoE/llava_image_tune_cleaned.json train_json/
if [ "$CLUSTER_NAME" == "FRCE" ]; then
    # frce
    mkdir $MYTMP_DIR/eval
    cd $MYTMP_DIR/eval && unzip -qq ${DATA_ROOT}/MoE/eval/eval.zip
    cd $MYTMP_DIR/eval/vqav2 && unzip -qq ${DATA_ROOT}/MoE/eval/vqav2/test2015.zip
    cd $MYTMP_DIR/eval/textvqa && unzip -qq ${DATA_ROOT}/MoE/eval/textvqa/train_val_images.zip
    cd $MYTMP_DIR/eval/textvqa && cp ${DATA_ROOT}/MoE//eval/textvqa/TextVQA_0.5.1_val.json .
    cd $MYTMP_DIR
else
    mkdir $MYTMP_DIR/eval
    cd $MYTMP_DIR/eval && unzip -qq ${DATA_ROOT}/eval/eval.zip
    cd $MYTMP_DIR/eval/vqav2 && unzip -qq ${DATA_ROOT}/eval/vqav2/test2015.zip
    cd $MYTMP_DIR/eval/textvqa && unzip -qq ${DATA_ROOT}/eval/textvqa/train_val_images.zip
    cd $MYTMP_DIR/eval/textvqa && cp ${DATA_ROOT}/eval/textvqa/TextVQA_0.5.1_val.json .
    cd $MYTMP_DIR
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
    atten_implementation=xformers    # no flash-attn
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
# model_name_or_path=Qwen/Qwen1.5-7B-Chat
# conv_version=qwen_1_5_v2
pretrain_output_dir=${DATA_ROOT}/temp_20240520/llava${MY_DEBUG}/${model_name_or_path}/llava-pretrain-${deepspeed_config}-${atten_implementation}-${LORA_POSTFIX}
finetune_output_dir=${pretrain_output_dir}/finetune_anyres
moe_output_dir=${finetune_output_dir}/moe
mkdir -p ${moe_output_dir}
NUM_CKPT_DIRS=$(find $finetune_output_dir -maxdepth 1 -type d -name "checkpoint-*" | wc -l)
LOG_FILE=$finetune_output_dir/log$((NUM_CKPT_DIRS + 1)).txt




cd $CODE_ROOT

EVAL="/lscratch/${SLURM_JOB_ID}/eval"

TYPE=${1}

if [ "${TYPE}" == "llama3" ]; then
CKPT_NAME=/data/zhongz2/data/LLaVA-Med/video/temp_20240520/llava/meta-llama/Meta-Llama-3-8B-Instruct/llava-pretrain-zero3-flash_attention_2-/finetune_anyres/llava_llama_debug
CKPT_NAME=/data/zhongz2/data/LLaVA-Med/video/temp_20240516/llava/meta-llama/Meta-Llama-3-8B-Instruct/llava-pretrain-zero3-flash_attention_2-/llava_llama3_debug
CKPT_NAME=/data/zhongz2/data/LLaVA-Med/video/temp_20240606/llava/meta-llama/Meta-Llama-3-8B-Instruct/llava-pretrain-zero3-flash_attention_2-/finetune_anyres/llava_llama3_debug
CONV="llava_llama_3_v2"
python -m llava.eval.model_vqa_loader_llama3 \
    --model-path ${CKPT_NAME} \
    --question-file ${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${EVAL}/textvqa/train_images \
    --answers-file ${EVAL}/textvqa/answers/${CKPT_NAME}-${CONV}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

else
CKPT_NAME=/data/zhongz2/data/LLaVA-Med/video/temp_20240520/llava/Qwen/Qwen1.5-7B-Chat/llava-pretrain-zero3-flash_attention_2-/finetune_anyres/llava_qwen_debug
CONV="qwen_1_5_v2"
python -m llava.eval.model_vqa_loader_qwen \
    --model-path ${CKPT_NAME} \
    --question-file ${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${EVAL}/textvqa/train_images \
    --answers-file ${EVAL}/textvqa/answers/${CKPT_NAME}-${CONV}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}
fi

python -m llava.eval.eval_textvqa \
    --annotation-file ${EVAL}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${EVAL}/textvqa/answers/${CKPT_NAME}-${CONV}.jsonl

rsync -avh ${EVAL}/textvqa/answers/${CKPT_NAME}-${CONV}.jsonl /data/zhongz2/temp29/





















