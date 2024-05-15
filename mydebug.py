import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava import conversation as conversation_lib
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
from llava.model import *

from PIL import Image


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def preprocess_multimodal(
    sources: Sequence[str]# ,
    # data_args: DataArguments
) -> Dict:
    # is_multimodal = data_args.is_multimodal
    # if not is_multimodal:
    #     return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if False: # "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if False: # data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

data_path = ['/lscratch/26382740/train_json/llava_image_tune_cleaned.json']
# data_path = ['/lscratch/26382740/train_json/nlp_tune.json']
data_path = ['/tmp/zhongz2/train_json/llava_image_tune_cleaned.json']


list_data_dict = []
for data in data_path:
    data = json.load(open(data, "r"))
    for i in data:
        i['id'] = len(list_data_dict)
        list_data_dict.append(i)

if True: # 
    model_name_or_path = "Qwen/Qwen1.5-7B-Chat"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir="./cache_dir",
        use_fast=False,
    )
    config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = LlavaQwenForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir="./cache_dir"
    )
    if False:
        prompt = "Give me a short introduction to large language model."
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    tokenizer.bos_token = '<|im_start|>'

    index = 0
    sources = [list_data_dict[index]]

    sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
    has_image = 'image' in list_data_dict[index]
    conv = copy.deepcopy(conversation_lib.conv_templates["qwen_1_5_v2"])
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            messages.append({'role': role, 'content': sentence["value"]})
        conversations.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ))
    
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN_V2

    # Mask targets
    for j, (conversation, target, input_id) in enumerate(zip(conversations, targets, input_ids)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        cur_len = 0 
        target[:] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            
            parts = rou.split(conv.sep2)
            rou_len = len(tokenizer_image_token(rou+conv.sep, tokenizer))
            # rou_len = len(tokenizer_image_token(rou+conv.sep if i!=len(rounds)-1 else rou, tokenizer))  # 
            if i!=0:
                # rou_len -= 1
                pass
            else:
                cur_len += rou_len
                continue

            ans_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            target[cur_len : cur_len + ans_len] = input_id[cur_len : cur_len + ans_len]

            cur_len += rou_len    

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )





if True:  # llava-llama3
    model = LlavaLlamaForCausalLM.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct',
        cache_dir="./cache_dir",
        attn_implementation=None,
        torch_dtype=torch.float16
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct',
        cache_dir='./cache_dir',
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = "<|reserved_special_token_0|>"

    if False:
        messages = [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": "Who are you?"},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

    for index in range(len(list_data_dict)):
        sources = [list_data_dict[index]]

        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
        has_image = 'image' in list_data_dict[index]
        conv = copy.deepcopy(conversation_lib.conv_templates["llava_llama_3_v2"])
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            messages = [{'role': 'system', 'content': conv.system}]
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                messages.append({'role': role, 'content': sentence["value"]})
            conversations.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                ))
        if has_image:
            input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        else:
            input_ids = tokenizer(
                conversations,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids
        targets = input_ids.clone()
        # chat_template_messages = [{"role": "system", "content": conv.system}]
        # for role, message in conv.messages:
        #     if message:
        #         if type(message) is tuple:
        #             message, images = message
        #             message = "<image>" * len(images) + message
        #         chat_template_messages.append({"role": role, "content": message})
        # prompt = tokenizer.apply_chat_template(chat_template_messages, tokenize=False, add_generation_prompt=True)

        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_V2

        # Mask targets 
        # sep = conv.sep + conv.roles[1] + ": "
        for j, (conversation, target, input_id) in enumerate(zip(conversations, targets, input_ids)):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            cur_len = 0 
            target[:] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                
                parts = rou.split(conv.sep2)
                rou_len = len(tokenizer_image_token(rou+conv.sep, tokenizer))  # if add_generation_prompt=True
                # rou_len = len(tokenizer_image_token(rou+conv.sep if i!=len(rounds)-1 else rou, tokenizer))  # 
                if i!=0:
                    rou_len -= 1
                else:
                    cur_len += rou_len
                    continue

                ans_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
                target[cur_len : cur_len + ans_len] = input_id[cur_len : cur_len + ans_len]

                cur_len += rou_len    

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )


if True:  # llava-llama2
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'lmsys/vicuna-7b-v1.5',
        cache_dir='./cache_dir',
        model_max_length=4096,
        padding_side="right",
        use_fast=False,
    )

    has_image = True
    tokenizer.pad_token = tokenizer.unk_token
    conv = copy.deepcopy(conversation_lib.conv_templates["vicuna_v1"])
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for j, (conversation, target) in enumerate(zip(conversations, targets)):
        print(f'begin conversion {j}', '='*80)
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            print(f'round {i}')
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            print(rou)
            print(tokenizer_image_token(rou, tokenizer))
            print(parts[0])
            print(tokenizer_image_token(parts[0], tokenizer))

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len

            print(cur_len)
            print(targets)

            print('\n')

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )



if False:
    # FastChat
    import transformers
    model_name_or_path = "lmsys/vicuna-7b-v1.5"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    import json
    train_json = json.load(open("data/dummy_conversation.json", "r"))
    sources = [example["conversations"] for example in train_json]
    sources = [sources[0]]

    from fastchat.conversation import SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template


    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())


    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
    IGNORE_TOKEN_ID = -100

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )









