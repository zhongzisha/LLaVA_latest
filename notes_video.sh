

if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th21_ds
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    CODE_ROOT=$FRCE_DATA_ROOT/LLaVA-NeXT
    MYTMP_DIR=/tmp/zhongz2
    DATA_ROOT=/mnt/gridftp/zhongz2
else
    source /data/zhongz2/anaconda3/bin/activate th21_ds
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0
    CODE_ROOT=/home/$USER/LLaVA-NeXT
    MYTMP_DIR=/lscratch/$SLURM_JOB_ID
    DATA_ROOT=/data/zhongz2/data/LLaVA-Med/video
fi
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH

cd $MYTMP_DIR
mkdir -p llava_video/video-chatgpt llava_video/video_detail_description
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip /data/zhongz2/data/LLaVA-Med/video/videochatgpt_test/Test_Videos.zip
unzip /data/zhongz2/data/LLaVA-Med/video/videochatgpt_test/Test_Human_Annotated_Captions.zip
unzip /data/zhongz2/data/LLaVA-Med/video/videochatgpt_test/Benchmarking_QA.zip
mv Test_Videos llava_video/video-chatgpt/
mv Benchmarking_QA/* llava_video/video-chatgpt/
mv Test_Human_Annotated_Captions llava_video/video_detail_description/

bash scripts/video/demo/video_demo1.sh lmms-lab/LLaVA-NeXT-Video-7B-DPO vicuna_v1 32 2 True ${MYTMP_DIR}/llava_video/video-chatgpt/Test_Videos/v_Lf_7RurLgp0.mp4
bash scripts/video/demo/video_demo1.sh lmms-lab/LLaVA-NeXT-Video-7B vicuna_v1 32 2 True ${MYTMP_DIR}/llava_video/video-chatgpt/Test_Videos/v_Lf_7RurLgp0.mp4

pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install sglang==0.1.12
pip install transformers --upgrade


# Evaluating Llama-3-LLaVA-NeXT-8B on multiple datasets
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --num_processes=2 \
  -m lmms_eval \
  --model llava \
  --model_args pretrained=lmms-lab/llama3-llava-next-8b,conv_template=llava_llama_3 \
  --tasks ai2d,chartqa,docvqa_val,mme,mmbench_en_dev \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_next \
  --output_path $MYTMP_DIR/logs/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 \
  -m lmms_eval \
  --model llava \
  --model_args pretrained=lmms-lab/llama3-llava-next-8b,conv_template=llava_llama_3,attn_implementation=eager \
  --tasks ai2d,chartqa,docvqa_val,mme,mmbench_en_dev \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_next \
  --output_path $MYTMP_DIR/logs/

# Evaluating LLaVA-NeXT-72B on multiple datasets
accelerate launch --num_processes=2 \
  -m lmms_eval \
  --model llava \
  --model_args pretrained=lmms-lab/llava-next-72b,conv_template=qwen_1_5,model_name=llava_qwen,device_map=auto \
  --tasks ai2d,chartqa,docvqa_val,mme,mmbench_en_dev \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_next \
  --output_path ./logs/
















LLM_VERSION="meta-llama/Meta-Llama-3-8B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

PROMPT_VERSION=plain
PRETRAIN_DATA_VERSION="blip558k"
############### Pretrain ################

BASE_RUN_NAME="llavanext-${LLM_VERSION_CLEAN}-${VISION_MODEL_VERSION_CLEAN}-pretrain_${PRETRAIN_DATA_VERSION}_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

PROMPT_VERSION="llava_llama_3"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-blip558k_pretrain_plain_la_1_6mix_ft"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

torchrun # with necessary torchrun information for distributed training\
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version $PROMPT_VERSION \
    --data_path="/path/to/data/llava_instruct/llava1_6mix.json" \
    --image_folder /path/to/data/llava_data \
    --pretrain_mm_mlp_adapter="./checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "./checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True













