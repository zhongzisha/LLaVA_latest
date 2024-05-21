source /data/zhongz2/anaconda3/bin/activate th21_ds
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
module load gcc/11.3.0
CODE_ROOT=/home/$USER/LLaVA
MYTMP_DIR=/lscratch/$SLURM_JOB_ID
DATA_ROOT=/data/zhongz2/data/LLaVA-Med/video
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH

cd $CODE_ROOT

EVAL="/lscratch/${SLURM_JOB_ID}/eval"
CONV="vicuna_v1"
CKPT_NAME=liuhaotian/llava-v1.5-13b
CKPT_NAME=liuhaotian/llava-v1.5-7b
# CONV=mistral_instruct
# CONV=mistral_direct
CKPT_NAME=liuhaotian/llava-v1.6-mistral-7b

CKPT_NAME=/data/zhongz2/data/LLaVA-Med/video/temp_20240404/llava/lmsys/vicuna-7b-v1.5/llava-pretrain-zero3-flash_attention_2-/llava-v1.7-7b
CONV="vicuna_v1"
CKPT_NAME=liuhaotian/llava-v1.5-7b
CKPT_NAME="lmms-lab/llama3-llava-next-8b"
python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_NAME} \
    --question-file ${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${EVAL}/textvqa/train_images \
    --answers-file ${EVAL}/textvqa/answers/${CKPT_NAME}-${CONV}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

CKPT_NAME=/data/zhongz2/data/LLaVA-Med/video/temp_20240518/llava/meta-llama/Meta-Llama-3-8B-Instruct/llava-pretrain-zero3-flash_attention_2-/llava_llama3_debug
# CKPT_NAME=/data/zhongz2/data/LLaVA-Med/video/temp_20240520/llava/meta-llama/Meta-Llama-3-8B-Instruct/llava-pretrain-zero3-flash_attention_2-/finetune_anyres/llava_llama_debug
CONV="llava_llama_3_v2"
# CKPT_NAME="lmms-lab/llama3-llava-next-8b"
python -m llava.eval.model_vqa_loader_llama3 \
    --model-path ${CKPT_NAME} \
    --question-file ${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${EVAL}/textvqa/train_images \
    --answers-file ${EVAL}/textvqa/answers/${CKPT_NAME}-${CONV}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python -m llava.eval.eval_textvqa \
    --annotation-file ${EVAL}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${EVAL}/textvqa/answers/${CKPT_NAME}-${CONV}.jsonl

CKPT_NAME=llava-v1.7-7b
CKPT=/data/zhongz2/data/LLaVA-Med/video/temp_20240404/llava/lmsys/vicuna-7b-v1.5/llava-pretrain-zero3-flash_attention_2-/${CKPT_NAME}
# CKPT=liuhaotian/llava-v1.5-7b
# CKPT_NAME=liuhaotian/llava-v1.5-7b
CONV="vicuna_v1"
CONV="llama_2"
DATASET_NAME=vqa_rad
CUDA_VISIBLE_DEVICES=1 python3 -m llava.eval.model_vqa_loader_med \
    --model-path ${CKPT} \
    --question-file /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/test.json \
    --image-folder /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/images \
    --answers-file /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/${CKPT}/${CONV}/answers/test-answer-file-run1.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

CUDA_VISIBLE_DEVICES=1 python llava/eval/run_eval_batch1.py \
--gt /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/test.json \
--candidate /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/train_open_answers.json \
--pred /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/${CKPT}/${CONV}/answers/test-answer-file-run1.jsonl \
--pred_file_parent_path /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/${CKPT}/${CONV}/answers/ \
--target_test_type test-answer-file-run1   



CKPT_NAME=llava-v1.7-8b
CKPT=/data/zhongz2/data/LLaVA-Med/video/temp_20240404/llava/lmsys/vicuna-7b-v1.5/llava-pretrain-zero3-flash_attention_2-/${CKPT_NAME}
# CKPT=liuhaotian/llava-v1.5-7b
# CKPT_NAME=liuhaotian/llava-v1.5-7b
CONV="vicuna_v1"
# CONV="llama_2"
DATASET_NAME=vqa_rad
CUDA_VISIBLE_DEVICES=1 python3 -m llava.eval.model_vqa_loader_med \
    --model-path ${CKPT} \
    --question-file /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/test.json \
    --image-folder /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/images \
    --answers-file /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/${CKPT}/${CONV}/answers/test-answer-file-run1.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

CUDA_VISIBLE_DEVICES=1 python llava/eval/run_eval_batch1.py \
--gt /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/test.json \
--candidate /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/train_open_answers.json \
--pred /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/${CKPT}/${CONV}/answers/test-answer-file-run1.jsonl \
--pred_file_parent_path /lscratch/$SLURM_JOB_ID/finetune_data_LLaVA-Med/${DATASET_NAME}/${CKPT}/${CONV}/answers/ \
--target_test_type test-answer-file-run1   



# 
rsync -avh --exclude "__pycache__" llava scripts $FRCE_SERVER:/scratch/cluster_scratch/zhongz2/LLaVA/












