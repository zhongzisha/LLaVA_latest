#!/bin/bash

echo "run job2"
echo `date`

if [ "$CLUSTER_NAME" == "FRCE" ]; then
    MYTMP_DIR=/tmp/zhongz2
    DATA_ROOT=/mnt/gridftp/zhongz2
else
    MYTMP_DIR=/lscratch/$SLURM_JOB_ID
    DATA_ROOT=/data/zhongz2/data/LLaVA-Med/video
fi

if [ ! -e /tmp/$USER ]; then
    mkdir -p /tmp/$USER 
fi
ln -sf $MYTMP_DIR /tmp/$USER/data
JSON_FOLDER="/tmp/$USER/data/train_json"
IMAGE_FOLDER="/tmp/$USER/data"
VIDEO_FOLDER="/tmp/$USER/data"

# JSON_FOLDER="/data/zhongz2/data/LLaVA-Med/video/MoE/train_json"
# IMAGE_FOLDER="/data/zhongz2/data/LLaVA-Med/video/MoE"
# VIDEO_FOLDER="/data/zhongz2/data/LLaVA-Med/video/MoE"

PRETRAIN_DATA="${JSON_FOLDER}/llava_image_.json ${JSON_FOLDER}/llava_med_alignment_500k_cleaned.json"
FINETUNE_DATA="${JSON_FOLDER}/llava_med_instruct_60k_cleaned.json ${JSON_FOLDER}/la_tune_256k.json ${JSON_FOLDER}/lrv_tune_331k.json ${JSON_FOLDER}/lvis_tune_220k_.json ${JSON_FOLDER}/svit_tune_157k.json ${JSON_FOLDER}/nlp_tune.json"
FINETUNE_DATA="${JSON_FOLDER}/llava_med_instruct_60k_cleaned.json ${JSON_FOLDER}/llava_image_tune_cleaned.json ${JSON_FOLDER}/nlp_tune.json"
FINETUNE_DATA="${JSON_FOLDER}/llava_med_instruct_60k_cleaned.json ${JSON_FOLDER}/llava_image_tune_cleaned.json"

MOE_FINETUNE_DATA="${JSON_FOLDER}/llava_med_instruct_60k_cleaned.json ${JSON_FOLDER}/llava_image_tune_cleaned.json ${JSON_FOLDER}/nlp_tune.json"

per_device_train_batch_size=${1}
gradient_accumulation_steps=${2}
learning_rate=${3}
data_type_str=${4}
deepspeed_config=${5}
atten_implementation=${6} 
model_name_or_path=${7}
pretrain_output_dir=${8}
finetune_output_dir=${9}
lora_params=${10}
conv_version=${11}

if [ -z "${MY_DEBUG}" ]; then
save_steps=1000
num_train_epochs=1
else
save_steps=2
num_train_epochs=0.005
fi

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
    --data_path ${FINETUNE_DATA} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${pretrain_output_dir}/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type "mlp2x_gelu" \
    --image_aspect_ratio anyres \
    --mm_patch_merge_type spatial_unpad \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
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
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ./cache_dir \
    --dataloader_drop_last True 

exit;



    --torch_compile True \
    --torch_compile_backend "inductor" \



torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \

deepspeed \
    llava/train/train_${atten_implementation}.py \
    ${lora_params} \
    --model_name_or_path ${model_name_or_path} \
    --version ${conv_version} \
    --data_path ${FINETUNE_DATA} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${pretrain_output_dir}/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type "mlp2x_gelu" \
    --image_aspect_ratio anyres \
    --mm_patch_merge_type spatial_unpad \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
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
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ./cache_dir \
    --dataloader_drop_last True \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'











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
    --model_name_or_path ${model_name_or_path} \
    --version ${conv_version} \
    --data_path ${FINETUNE_DATA} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${pretrain_output_dir}/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type "mlp2x_gelu" \
    --image_aspect_ratio anyres \
    --mm_patch_merge_type spatial_unpad \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
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
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ./cache_dir \
    --dataloader_drop_last True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'


# Qwen1.5
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
    --model_name_or_path ${model_name_or_path} \
    --version ${conv_version} \
    --data_path ${FINETUNE_DATA} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${pretrain_output_dir}/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type "mlp2x_gelu" \
    --image_aspect_ratio anyres \
    --mm_patch_merge_type spatial_unpad \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
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
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ./cache_dir \
    --dataloader_drop_last True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer'











