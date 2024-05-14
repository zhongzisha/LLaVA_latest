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

data_path = ['/lscratch/26308749/train_json/llava_image_tune_cleaned.json']
data_path = ['/lscratch/26308749/train_json/nlp_tune.json']

list_data_dict = []
for data in data_path:
    data = json.load(open(data, "r"))
    for i in data:
        i['id'] = len(list_data_dict)
        list_data_dict.append(i)

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

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
        IMAGE_TOKEN_INDEX = 128256
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt', image_token_index=IMAGE_TOKEN_INDEX) for prompt in conversations], dim=0)

        # chat_template_messages = [{"role": "system", "content": conv.system}]
        # for role, message in conv.messages:
        #     if message:
        #         if type(message) is tuple:
        #             message, images = message
        #             message = "<image>" * len(images) + message
        #         chat_template_messages.append({"role": role, "content": message})
        # prompt = tokenizer.apply_chat_template(chat_template_messages, tokenize=False, add_generation_prompt=True)

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

        # Mask targets
        sep2 = '<|start_header_id|>user<|end_header_id|>\n\n'
        sep = '<|start_header_id|>assistant<|end_header_id|>\n\n'
        # sep = conv.sep + conv.roles[1] + ": "
        for j, (conversation, target, input_id) in enumerate(zip(conversations, targets, input_ids)):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(sep)
            cur_len = 0 
            target[:] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                
                parts = rou.split(sep2)
                rou_len = len(tokenizer_image_token(rou+sep, tokenizer, image_token_index=IMAGE_TOKEN_INDEX))
                if i!=0:
                    rou_len -= 1
                else:
                    cur_len += rou_len
                    continue

                ans_len = len(tokenizer_image_token(parts[0], tokenizer, image_token_index=IMAGE_TOKEN_INDEX)) - 1
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











