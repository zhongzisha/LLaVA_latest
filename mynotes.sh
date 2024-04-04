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
CONV=mistral_instruct
CONV=mistral_direct
CKPT_NAME=liuhaotian/llava-v1.6-mistral-7b
python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_NAME} \
    --question-file ${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${EVAL}/textvqa/train_images \
    --answers-file ${EVAL}/textvqa/answers/${CKPT_NAME}-${CONV}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python -m llava.eval.eval_textvqa \
    --annotation-file ${EVAL}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${EVAL}/textvqa/answers/${CKPT_NAME}-${CONV}.jsonl




















