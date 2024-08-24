import sys,os
import argparse
import torch

from debug3 import DebugLlavaForCausalLM, load_sharded_checkpoint, tokenizer_image_token, process_anyres_image, SeparatorStyle, Conversation
from common import WORKER_HEART_BEAT_INTERVAL, build_logger, server_error_msg, pretty_print_semaphore, \
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer, AutoConfig
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def load_pretrained_model(gpu_id=-1):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cache_dir = '/Users/zhongz2/down/cache_dir'
    model_name_or_path = '/Users/zhongz2/down/finetune_llama_3_1_with_pretrain'
    cache_dir = './cache_dir'
    model_name_or_path = '/data/zhongz2/temp29/output_llava_llama_3/pretrain_anyres_debug3/finetune_llama_3_1_with_pretrain'
    conv_version = 'llama_3_1'
    eot_str = "<|eot_id|>"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print('gpu_id', gpu_id)
    device = torch.device(f"cuda:{gpu_id}") if gpu_id>=0 else torch.device("cpu")
    kwargs = {
        "device_map": device,
        "torch_dtype": torch.float16
    }
    cfg_pretrained = AutoConfig.from_pretrained(model_name_or_path)
    if conv_version in ['llama_3', 'llama_3_1']:
        model = DebugLlavaForCausalLM.from_pretrained(model_name_or_path, config=cfg_pretrained, attn_implementation="eager", **kwargs)
    elif conv_version == 'gemma_2':
        model = DebugLlavaGemma2ForCausalLM.from_pretrained(model_name_or_path, config=cfg_pretrained, attn_implementation="eager", **kwargs)
    elif conv_version == 'qwen_2':
        model = DebugLlavaQwen2ForCausalLM.from_pretrained(model_name_or_path, config=cfg_pretrained, attn_implementation="flash_attention_2", **kwargs)
    model.initialize_vision_modules(device=device, dtype=torch.float16)
    load_sharded_checkpoint(model, model_name_or_path)
    model.to(device)
    model.eval()
    return tokenizer, model, model.image_processor 
    

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def main(args):
    # Model
    # disable_torch_init()
    context_len = 2048
    tokenizer, model, image_processor = load_pretrained_model(gpu_id=args.gpu_id)


    conv = Conversation(
        system="You are a pirate chatbot who always responds in pirate speak!",
        roles=("user", "assistant"),
        version="llama_3_1",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.LLAMA_3_1,
        stop_token_ids=[128009],
        sep='<|start_header_id|>assistant<|end_header_id|>\n\n',
        sep2='<|start_header_id|>user<|end_header_id|>\n\n'
    )
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    messages = [{'role': 'system', 'content': conv.system}]

    image = load_image(args.image_file)
    image_size = image.size
    # Similar operation in model_worker.py 
    image_tensor = process_anyres_image(image, image_processor, model.config.image_grid_pinpoints)
    print('image_tensor', image_tensor.shape)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{conv.roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{conv.roles[1]}: ", end="")

        if image is not None:
            # first message
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None
        

        messages.append({'role': conv.roles[0], 'content': inp})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0]).strip()
        messages.append({'role': conv.roles[1], 'content': outputs})

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", type=str, default="./example.png")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
