
from typing import List, Optional, Tuple, Union, Any, Dict, Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
import os
import copy
import json
import base64
import io
import numpy as np
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import pathlib
import ast
import math
import re
import random
import shortuuid

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
from transformers import LlamaModel, LlamaForCausalLM, AutoTokenizer 
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

USE_TRANSFORMERS_TRAINER = True
if USE_TRANSFORMERS_TRAINER:
    from transformers.trainer import has_length, is_sagemaker_mp_enabled, \
        PreTrainedModel, TRAINING_ARGS_NAME, \
        SAFE_WEIGHTS_NAME, WEIGHTS_NAME
    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp
else: # using deepspeed (out of memory)
    from torch.utils.data import DataLoader
    import deepspeed
    from transformers import AdamW, get_scheduler

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


@dataclass
class DataArguments:
    data_path: Optional[List[str]] = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_crop_resolution: int = 224
    image_grid_pinpoints: Optional[str] = field(default="[]")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    conv_version: Optional[str] = field(default="plain")
    pretrain_ckpt_path: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    group_by_modality_length: bool = field(default=False)    
    remove_unused_columns: bool = field(default=False)
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    PLAIN = auto()
    LLAMA_3 = auto()


@dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


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


def preprocess_plain(
    sources: Sequence[str], 
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN 
        conversations.append(source[0]['value'] + source[1]['value'] + '\n')
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    if input_ids[0][0] != tokenizer.bos_token_id:
        input_ids = [torch.cat([torch.LongTensor([tokenizer.bos_token_id]), i]) for i in input_ids]
        targets = [torch.cat([torch.LongTensor([IGNORE_INDEX]), i]) for i in targets]
    
    return dict(input_ids=input_ids, labels=targets)


def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    conversation: Conversation,
    has_image: bool = False
) -> Dict:
    conv = conversation
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
                
    if input_ids[0][0] != tokenizer.bos_token_id:
        input_ids = [torch.cat([torch.LongTensor([tokenizer.bos_token_id]), i]) for i in input_ids]
        targets = [torch.cat([torch.LongTensor([IGNORE_INDEX]), i]) for i in targets]

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    conversation: Conversation,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation.sep_style == SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation.sep_style == SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, conversation=conversation, has_image=has_image)

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

    return sources



def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    # if isinstance(grid_pinpoints, str):
    #     assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
    #     grid_pinpoints = grid_pinpoints.replace(" ", "").replace("x", ",")[1:-1].split("),(")
    #     grid_pinpoints = [[int(x) * patch_size for x in item.split(",")] for item in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size

def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 conversation: Conversation,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        # ================================================
        list_data_dict = []
        for data in data_args.data_path:
            data = json.load(open(data, "r"))
            for i in data:
                i['id'] = len(list_data_dict)
                list_data_dict.append(i)
        # ================================================
        self.tokenizer = tokenizer
        self.conversation = conversation
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image'] 
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(self.data_args.image_folder, image_file)).convert('RGB')
            image_size = image.size
            max_size = max(image_size)
            if max_size > 1536:
                scale = 1024. / max_size
                image_size = (int(scale*image_size[0]), int(scale*image_size[1]))
                image = image.resize(image_size)
            
            if self.data_args.image_aspect_ratio == 'anyres':
                image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints)
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        data_dict = preprocess(sources, self.tokenizer, self.conversation, has_image=('image' in self.list_data_dict[i]))

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size 

        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['width'], crop_size['height']) 

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            image_sizes = [instance['image_size'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            batch['image_sizes'] = image_sizes

        return batch



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                conversation: Conversation,
                                data_args: DataArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, conversation=conversation, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)




def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


class LLaVATrainer(transformers.Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                if self.args.mm_vision_tower_lr is not None:
                    vision_tower_parameters = [name for name, _ in opt_model.named_parameters() if "vision_tower" in name]
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in vision_tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n in vision_tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_vision_tower_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in vision_tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n in vision_tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_vision_tower_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
                else:
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'conv_version', 'plain') == 'plain':
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler'] 

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'conv_version', 'plain') == 'plain':
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def _save1(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        supported_classes = (PreTrainedModel,) # if not is_peft_available() else (PreTrainedModel, PeftModel)
        print('supported_classes', supported_classes)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                print('using accelerator.unwrap_model')
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                print('not using accelerator.unwrap_model')
                if self.args.save_safetensors:
                    print('safetensors')
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    print('not safetensors')
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            # print('not supported classes')
            # self.model.save_pretrained(
            #     output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            # )
            print('using accelerator.unwrap_model')
            self.accelerator.unwrap_model(self.model).save_pretrained(
                output_dir, safe_serialization=self.args.save_safetensors,
                state_dict=self.accelerator.get_state_dict(self.model),
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
            )

        if self.tokenizer is not None:
            print('saving tokenizer')
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        print('saving training args')
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class DebugLlavaConfig(LlamaConfig):
    model_type = "debug_llava"
    temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_p: Optional[float] = None
    rope_scaling: Optional[dict] = {}


class DebugLlavaForCausalLM(LlamaForCausalLM):
    config_class = DebugLlavaConfig

    def __init__(self, config):
        super().__init__(config) 

    def initialize_vision_modules(self, device="auto", dtype=torch.bfloat16, mm_projector_type="mlp2x_gelu", pretrain_ckpt_path=None):
        vision_tower_name_or_ckpt = 'openai/clip-vit-large-patch14-336'
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name_or_ckpt)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name_or_ckpt).to(device=device, dtype=dtype)

        if mm_projector_type == 'linear':
            self.mm_projector = nn.Linear(self.vision_tower.config.hidden_size, self.config.hidden_size, device=device, dtype=dtype)
        elif mm_projector_type == 'mlp2x_gelu':
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", mm_projector_type)
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(self.vision_tower.config.hidden_size, self.config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
            self.mm_projector = nn.Sequential(*modules)

        if pretrain_ckpt_path is not None:
            print('loading pretrain ckpt path for mm_projector')
            self.mm_projector.load_state_dict(torch.load(pretrain_ckpt_path), strict=False)
        self.mm_projector.to(device=device, dtype=dtype)

        self.num_patches_per_side = self.vision_tower.config.image_size // self.vision_tower.config.patch_size

        embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=dtype))
        self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=dtype, device=device) * embed_std)

    def encode_images(self, images):
        vision_tower_outputs = self.vision_tower(images, output_hidden_states=True) 
        image_features = vision_tower_outputs.hidden_states[-2][:, 1:]
        image_features = self.mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None):
        if images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:

            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)

            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                # FIXME: now assume the image is square, and split to 2x2 patches
                # num_patches = h * w, where h = w = sqrt(num_patches)
                # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                # we want to first unflatten it to (2, 2, h, w, hidden_size)

                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.num_patches_per_side
                    assert height * width == base_image_feature.shape[0]
                    
                    if self.config.image_aspect_ratio == "anyres":
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.vision_tower.config.image_size)
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        image_feature = image_feature.view(2, 2, height, width, -1)

                    # spatial_unpad
                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                    image_feature = unpad_image(image_feature, image_sizes[image_idx])
                    image_feature = torch.cat((image_feature, self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)

                new_image_features.append(image_feature)

            image_features = new_image_features

        else:
            image_features = self.encode_images(images)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask  FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.model.embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.model.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = \
                self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes)
        else:
            inputs_embeds = self.model.embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("debug_llava", DebugLlavaConfig)
AutoModelForCausalLM.register(DebugLlavaConfig, DebugLlavaForCausalLM)


def debug():
    batch_size = 2
    images = []
    for i in range(batch_size):
        images.append(Image.fromarray(np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)))

    device_map = "auto"
    vision_tower_name_or_ckpt = 'openai/clip-vit-large-patch14-336'
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name_or_ckpt)
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name_or_ckpt, device_map=device_map)

    device = vision_tower.device

    inputs = image_processor(images=images, return_tensors="pt")
    # dict_keys(['pixel_values'])

    image_forward_outs = vision_tower(inputs['pixel_values'].to(device=device), output_hidden_states=True)
    # dict_keys(['last_hidden_state', 'pooler_output', 'hidden_states'])

    image_features = image_forward_outs.hidden_states[-2][:, 1:]
    # torch.Size([2, 576, 1024])

    hidden_size = vision_tower.config.hidden_size
    # 1024

    mm_projector = nn.Linear(hidden_size, hidden_size)
    mm_projector.to(device)

    image_features_projected = mm_projector(image_features)



def trainer_save_model_safe_fsdp(trainer: transformers.Trainer,
                                 output_dir: str):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model(output_dir=output_dir)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, 'conv_version', 'plain') == 'plain':
        # Only save Adapter
        keys_to_match = ['mm_projector']

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa




def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    if True: # is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
        if deterministic:
            torch.use_deterministic_algorithms(True)
    # if is_torch_mlu_available():
    #     torch.mlu.manual_seed_all(seed)
    # if is_torch_npu_available():
    #     torch.npu.manual_seed_all(seed)
    # if is_torch_xpu_available():
    #     torch.xpu.manual_seed_all(seed)
    # if is_tf_available():
    #     import tensorflow as tf

    #     tf.random.set_seed(seed)
    #     if deterministic:
    #         tf.config.experimental.enable_op_determinism()

def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result




class HfDeepSpeedConfig:
    """
    This object contains a DeepSpeed configuration dictionary and can be quickly queried for things like zero stage.

    A `weakref` of this object is stored in the module's globals to be able to access the config from areas where
    things like the Trainer object is not available (e.g. `from_pretrained` and `_get_resized_embeddings`). Therefore
    it's important that this object remains alive while the program is still running.

    [`Trainer`] uses the `HfTrainerDeepSpeedConfig` subclass instead. That subclass has logic to sync the configuration
    with values of [`TrainingArguments`] by replacing special placeholder values: `"auto"`. Without this special logic
    the DeepSpeed configuration is not modified in any way.

    Args:
        config_file_or_dict (`Union[str, Dict]`): path to DeepSpeed config file or dict.

    """

    def __init__(self, config_file_or_dict):
        if isinstance(config_file_or_dict, dict):
            # Don't modify user's data should they want to reuse it (e.g. in tests), because once we
            # modified it, it will not be accepted here again, since `auto` values would have been overridden
            config = copy.deepcopy(config_file_or_dict)
        elif os.path.exists(config_file_or_dict):
            with open(config_file_or_dict, encoding="utf-8") as f:
                config = json.load(f)
        else:
            try:
                config_decoded = base64.urlsafe_b64decode(config_file_or_dict).decode("utf-8")
                config = json.loads(config_decoded)
            except (UnicodeDecodeError, AttributeError, ValueError):
                raise ValueError(
                    f"Expected a string path to an existing deepspeed config, or a dictionary, or a base64 encoded string. Received: {config_file_or_dict}"
                )

        self.config = config

        self.set_stage_and_offload()

    def set_stage_and_offload(self):
        # zero stage - this is done as early as possible, before model is created, to allow
        # ``is_deepspeed_zero3_enabled`` query and getting to the early deepspeed config object
        # during ``zero.Init()`` which needs to know the dtype, and some other hparams.
        self._stage = self.get_value("zero_optimization.stage", -1)

        # offload
        self._offload = False
        if self.is_zero2() or self.is_zero3():
            offload_devices_valid = set(["cpu", "nvme"])
            offload_devices = set(
                [
                    self.get_value("zero_optimization.offload_optimizer.device"),
                    self.get_value("zero_optimization.offload_param.device"),
                ]
            )
            if len(offload_devices & offload_devices_valid) > 0:
                self._offload = True

    def find_config_node(self, ds_key_long):
        config = self.config

        # find the config node of interest if it exists
        nodes = ds_key_long.split(".")
        ds_key = nodes.pop()
        for node in nodes:
            config = config.get(node)
            if config is None:
                return None, ds_key

        return config, ds_key

    def get_value(self, ds_key_long, default=None):
        """
        Returns the set value or `default` if no value is set
        """
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return default
        return config.get(ds_key, default)

    def del_config_sub_tree(self, ds_key_long, must_exist=False):
        """
        Deletes a sub-section of the config file if it's found.

        Unless `must_exist` is `True` the section doesn't have to exist.
        """
        config = self.config

        # find the config node of interest if it exists
        nodes = ds_key_long.split(".")
        for node in nodes:
            parent_config = config
            config = config.get(node)
            if config is None:
                if must_exist:
                    raise ValueError(f"Can't find {ds_key_long} entry in the config: {self.config}")
                else:
                    return

        # if found remove it
        if parent_config is not None:
            parent_config.pop(node)

    def is_true(self, ds_key_long):
        """
        Returns `True`/``False` only if the value is set, always `False` otherwise. So use this method to ask the very
        specific question of whether the value is set to `True` (and it's not set to `False`` or isn't set).

        """
        value = self.get_value(ds_key_long)
        return False if value is None else bool(value)

    def is_false(self, ds_key_long):
        """
        Returns `True`/``False` only if the value is set, always `False` otherwise. So use this method to ask the very
        specific question of whether the value is set to `False` (and it's not set to `True`` or isn't set).
        """
        value = self.get_value(ds_key_long)
        return False if value is None else not bool(value)

    def is_zero2(self):
        return self._stage == 2

    def is_zero3(self):
        return self._stage == 3

    def is_offload(self):
        return self._offload


from functools import partialmethod
class HfTrainerDeepSpeedConfig(HfDeepSpeedConfig):
    """
    The `HfTrainerDeepSpeedConfig` object is meant to be created during `TrainingArguments` object creation and has the
    same lifespan as the latter.
    """

    def __init__(self, config_file_or_dict):
        super().__init__(config_file_or_dict)
        self._dtype = None
        self.mismatches = []

    def dtype(self):
        if self._dtype is None:
            raise ValueError("trainer_config_process() wasn't called yet to tell dtype")
        return self._dtype

    def is_auto(self, ds_key_long):
        val = self.get_value(ds_key_long)
        if val is None:
            return False
        else:
            return val == "auto"

    def fill_match(self, ds_key_long, hf_val, hf_key=None, must_match=True):
        """
        A utility method that massages the config file and can optionally verify that the values match.

        1. Replace "auto" values with `TrainingArguments` value.

        2. If it wasn't "auto" and `must_match` is true, then check that DS config matches Trainer
        config values and if mismatched add the entry to `self.mismatched` - will assert during
        `trainer_config_finalize` for one or more mismatches.

        """
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return

        if config.get(ds_key) == "auto":
            config[ds_key] = hf_val
            return

        if not must_match:
            return

        ds_val = config.get(ds_key)
        if ds_val is not None and ds_val != hf_val:
            self.mismatches.append(f"- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}")

    fill_only = partialmethod(fill_match, must_match=False)

    def trainer_config_process(self, args, auto_find_batch_size=False):
        """
        Adjust the config with `TrainingArguments` values. This stage is run during `TrainingArguments` object
        creation.
        """
        # DeepSpeed does:
        # train_batch_size = world_size * train_micro_batch_size_per_gpu * gradient_accumulation_steps
        train_batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.fill_match(
            "train_micro_batch_size_per_gpu",
            args.per_device_train_batch_size,
            "per_device_train_batch_size",
            not auto_find_batch_size,
        )
        self.fill_match(
            "gradient_accumulation_steps",
            args.gradient_accumulation_steps,
            "gradient_accumulation_steps",
        )
        self.fill_match(
            "train_batch_size",
            train_batch_size,
            "train_batch_size (calculated)",
            not auto_find_batch_size,
        )
        self.fill_match("gradient_clipping", args.max_grad_norm, "max_grad_norm")

        self.fill_match("optimizer.params.lr", args.learning_rate, "learning_rate")
        self.fill_match(
            "optimizer.params.betas",
            [args.adam_beta1, args.adam_beta2],
            "adam_beta1+adam_beta2",
        )
        self.fill_match("optimizer.params.eps", args.adam_epsilon, "adam_epsilon")
        self.fill_match("optimizer.params.weight_decay", args.weight_decay, "weight_decay")

        self.fill_only("scheduler.params.warmup_min_lr", 0)  # not a trainer arg
        self.fill_match("scheduler.params.warmup_max_lr", args.learning_rate, "learning_rate")
        # total_num_steps - will get set in trainer_config_finalize

        # fp16
        if args.fp16 or args.fp16_full_eval:
            fp16_backend = "apex" if args.fp16_backend == "apex" else "amp"
        else:
            fp16_backend = None

        if args.save_on_each_node:
            # deepspeed uses shared storage by default. Let's override this setting if save_on_each_node == True
            self.config["checkpoint"] = self.config.get("checkpoint", {})
            self.config["checkpoint"]["use_node_local_storage"] = args.save_on_each_node

        # amp: similar to the pytorch native amp - it has a bunch of optional params but we won't set
        # any here unless the user did the work
        self.fill_match(
            "fp16.enabled",
            ((args.fp16 or args.fp16_full_eval) and fp16_backend == "amp"),
            "fp16|fp16_full_eval+fp16_backend(amp)",
        )

        # apex: delegates amp work to apex (which needs to be available), but it cannot be used with any
        # ZeRO features
        self.fill_match("amp.enabled", fp16_backend == "apex", "fp16+fp16_backend(apex)")
        self.fill_match("amp.opt_level", args.fp16_opt_level, "fp16_opt_level")

        self.fill_match("bf16.enabled", (args.bf16 or args.bf16_full_eval), "bf16|bf16_full_eval")

        # deepspeed's default mode is fp16 unless there is a config that says differently
        if self.is_true("bf16.enabled"):
            self._dtype = torch.bfloat16
        elif self.is_false("fp16.enabled"):
            self._dtype = torch.float32
        else:
            self._dtype = torch.float16

    def trainer_config_finalize(self, args, model, num_training_steps):
        """
        This stage is run after we have the model and know num_training_steps.

        Now we can complete the configuration process.
        """
        # zero

        # deal with config keys that use `auto` value and rely on model's hidden_size
        hidden_size_based_keys = [
            "zero_optimization.reduce_bucket_size",
            "zero_optimization.stage3_prefetch_bucket_size",
            "zero_optimization.stage3_param_persistence_threshold",
        ]
        hidden_size_auto_keys = [x for x in hidden_size_based_keys if self.is_auto(x)]

        if len(hidden_size_auto_keys) > 0:
            if hasattr(model.config, "hidden_size"):
                hidden_size = model.config.hidden_size
            elif hasattr(model.config, "hidden_sizes"):
                # if there are many hidden sizes pick the largest one
                hidden_size = max(model.config.hidden_sizes)
            else:
                raise ValueError(
                    "The model's config file has neither `hidden_size` nor `hidden_sizes` entry, "
                    "therefore it's not possible to automatically fill out the following `auto` entries "
                    f"in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing "
                    "`auto` values for these keys with an integer value of your choice."
                )

            self.fill_only("zero_optimization.reduce_bucket_size", hidden_size * hidden_size)
            if self.is_zero3():
                # automatically assign the optimal config values based on model config
                self.fill_only(
                    "zero_optimization.stage3_prefetch_bucket_size",
                    0.9 * hidden_size * hidden_size,
                )
                self.fill_only(
                    "zero_optimization.stage3_param_persistence_threshold",
                    10 * hidden_size,
                )

        # scheduler
        self.fill_match(
            "scheduler.params.total_num_steps",
            num_training_steps,
            "num_training_steps (calculated)",
        )
        self.fill_match(
            "scheduler.params.warmup_num_steps",
            args.get_warmup_steps(num_training_steps),
            "warmup_steps",
        )

        if len(self.mismatches) > 0:
            mismatches = "\n".join(self.mismatches)
            raise ValueError(
                "Please correct the following DeepSpeed config values that mismatch TrainingArguments"
                f" values:\n{mismatches}\nThe easiest method is to set these DeepSpeed config values to 'auto'."
            )



def train_with_deepspeed():
    model_name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'

    parser = transformers.HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    print('local_rank', local_rank)
    if local_rank == 0:
        print(data_args)
        print(model_args)
        print(training_args)
        print(training_args.train_batch_size)

    if local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)
        random.seed(training_args.seed)
        np.random.seed(training_args.seed)
        torch.manual_seed(training_args.seed)
        torch.cuda.manual_seed_all(training_args.seed)
    torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.unk_token = "<|reserved_special_token_0|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.conv_version == 'plain':  # for pretrain
        conv_llava = Conversation(
            system="",
            roles=("", ""),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.PLAIN,
            sep="\n",
        )
    elif model_args.conv_version == 'llama_3': # for fine tune
        conv_llava = Conversation(
            system="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
            roles=("user", "assistant"),
            version="llama_v3",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.LLAMA_3,
            stop_token_ids=[128009],
            sep='<|start_header_id|>assistant<|end_header_id|>\n\n',
            sep2='<|start_header_id|>user<|end_header_id|>\n\n'
        )
    else:
        raise ValueError("wrong conv_version")

    model = DebugLlavaForCausalLM.from_pretrained(model_name_or_path, cache_dir=training_args.cache_dir, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    model.initialize_vision_modules(device=training_args.device, dtype=torch.bfloat16, pretrain_ckpt_path=model_args.pretrain_ckpt_path)
    model.to(training_args.device)

    training_args.conv_version = model_args.conv_version
    if model_args.conv_version == 'plain': 
        # self.model.requires_grad_(False)
        # self.lm_head.requires_grad_(False)
        # self.vision_tower.requires_grad_(False)
        # self.image_newline.requires_grad_(False)
        model.requires_grad_(False)
        for p in model.mm_projector.parameters():
            p.requires_grad = True
    else:
        # model.vision_tower.requires_grad_(False)
        lr_of_vit = training_args.mm_vision_tower_lr if training_args.mm_vision_tower_lr is not None else training_args.learning_rate
        lr_of_mlp = training_args.mm_projector_lr if training_args.mm_projector_lr is not None else training_args.learning_rate
        training_args.mm_projector_lr = lr_of_mlp
    
    if local_rank == 0:
        # Calculate total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('total_params: ', total_params)
        print('trainable_params: ', trainable_params)
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name, p.shape)

    if training_args.gradient_checkpointing:
        print('enabled gradient_checkpointing')
        
        from functools import partial
        notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        torch.utils.checkpoint.checkpoint = notfailing_checkpoint

        # training_args.gradient_checkpointing_kwargs=dict(use_reentrant=False)
        # if hasattr(model, "enable_input_require_grads"):
        #     model.enable_input_require_grads()
        # else:
        #     def make_inputs_require_grad(module, input, output):
        #         output.requires_grad_(True)
        #     model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if training_args.gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        else:
            gradient_checkpointing_kwargs = training_args.gradient_checkpointing_kwargs
        gradient_checkpointing_kwargs.update({"use_reentrant": False})
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)


    data_args.image_processor = model.image_processor
    data_args.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer, conversation=conv_llava, data_args=data_args)
    print('len train dataset', len(data_module['train_dataset']))

    model.config.use_cache = False
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    if data_args.image_aspect_ratio == 'anyres':
        base_size = model.vision_tower.config.image_size
        grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3]]
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints = [
            [g[0]*base_size, g[1]*base_size] for g in grids]
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr

    # build dataloader
    dataloader_params = {
        "batch_size": training_args.train_batch_size,
        "collate_fn": data_module['data_collator'],
        "num_workers": training_args.dataloader_num_workers,
        "pin_memory": training_args.dataloader_pin_memory,
        "persistent_workers": training_args.dataloader_persistent_workers,
    }

    if not isinstance(data_module['train_dataset'], torch.utils.data.IterableDataset): 
        dataloader_params["sampler"] = LengthGroupedSampler(
            training_args.train_batch_size,
            world_size=training_args.world_size * training_args.gradient_accumulation_steps,
            lengths=data_module['train_dataset'].modality_lengths,
            group_by_modality=True,
        )
        dataloader_params["drop_last"] = training_args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker
        dataloader_params["prefetch_factor"] = training_args.dataloader_prefetch_factor

    train_dataloader = DataLoader(data_module['train_dataset'], **dataloader_params)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch) 
    num_train_epochs = math.ceil(training_args.num_train_epochs)

    def to_device(batch):
        output = {}
        for k, v in batch.items():
            try:
                output[k] = v.to(device)
            except:
                output[k] = v
        return output

    # build optimizer
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    if training_args.mm_projector_lr is not None:
        projector_parameters = [name for name, _ in model.named_parameters() if "mm_projector" in name]
        if training_args.mm_vision_tower_lr is not None:
            vision_tower_parameters = [name for name, _ in model.named_parameters() if "vision_tower" in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in vision_tower_parameters and p.requires_grad)
                    ],
                    "weight_decay": training_args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n in vision_tower_parameters and p.requires_grad)
                    ],
                    "weight_decay": training_args.weight_decay,
                    "lr": training_args.mm_vision_tower_lr,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in vision_tower_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n in vision_tower_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": training_args.mm_vision_tower_lr,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": training_args.weight_decay,
                    "lr": training_args.mm_projector_lr,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": training_args.mm_projector_lr,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": training_args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": training_args.weight_decay,
                    "lr": training_args.mm_projector_lr,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": training_args.mm_projector_lr,
                },
            ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    world_size = torch.distributed.get_world_size()
    if training_args.warmup_ratio is not None:
        training_args.num_warmup_steps = int(training_args.max_train_steps*training_args.warmup_ratio)
    else:
        print('warmup_ratio is None')
    print('training_args.num_warmup_steps', training_args.num_warmup_steps)
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=training_args.max_train_steps,
    )

    hf_deepspeed_config = HfTrainerDeepSpeedConfig(training_args.deepspeed)
    hf_deepspeed_config.trainer_config_process(training_args)
    print('hf_deepspeed_config', hf_deepspeed_config.config)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=hf_deepspeed_config.config,
        lr_scheduler=lr_scheduler,
        dist_init_required=False)


    # Train
    micro_step = 0
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = to_device(batch)                
            outputs = model(**batch)
            loss = outputs.loss
            # loss = loss / args.gradient_accumulation_steps # DeepSpeed engine will handle this loss scaling (_scale_loss_by_gas), thus no need to do so on user side
            model.backward(loss)
            model.step()
            micro_step += 1
            if micro_step % training_args.gradient_accumulation_steps == 0:
                global_step += 1
        if training_args.output_dir is not None:
            if not os.path.isdir(training_args.output_dir):
                os.makedirs(training_args.output_dir)

            if torch.distributed.get_rank() == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                CONFIG_NAME = "config.json"
                WEIGHTS_NAME = "pytorch_model.bin"
                output_model_file = os.path.join(training_args.output_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(training_args.output_dir, CONFIG_NAME)
                torch.save(save_without_random_ltd(model_to_save), output_model_file)
                output_config_file = os.path.join(training_args.output_dir, CONFIG_NAME)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(training_args.output_dir)

def eval():
    model_name_or_path = '/data/zhongz2/temp29/output_llava_llama_3/pretrain_anyres_debug3/finetune'
    cache_dir = '/data/zhongz2/data/cache_dir'
    model_name = 'llava_llama_3_debug'

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False
    )

    device = torch.device("cuda:0")
    kwargs = {
        "device_map": "cuda:0",
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.float16
    }
    cfg_pretrained = AutoConfig.from_pretrained(model_name_or_path)
    model = DebugLlavaForCausalLM.from_pretrained(model_name_or_path, config=cfg_pretrained, **kwargs)
    model.initialize_vision_modules(device=device, dtype=torch.float16)
    model.to(device)
    model.eval()
    from transformers.modeling_utils import load_sharded_checkpoint
    load_sharded_checkpoint(model, model_name_or_path)

    # Custom dataset class
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
            self.questions = questions
            self.image_folder = image_folder
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self.model_config = model_config
            system = "You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language."
            self.input_ids = []
            for index in range(len(self.questions)):
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': DEFAULT_IMAGE_TOKEN + '\n' + self.questions[index]['text']}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                )
                if index < 5:
                    print(prompt)
                self.input_ids.append(
                    tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                )

        def __getitem__(self, index): 
            image = Image.open(os.path.join(self.image_folder, self.questions[index]["image"])).convert('RGB')
            image_size = image.size
            max_size = max(image_size)
            if max_size > 1536:
                scale = 1024. / max_size
                image_size = (int(scale*image_size[0]), int(scale*image_size[1]))
                image = image.resize(image_size)
            image_tensor = process_anyres_image(image, model.image_processor, model.config.image_grid_pinpoints)
            return self.input_ids[index], image_tensor, image_size

        def __len__(self):
            return len(self.questions)


    def collate_fn(batch):
        input_ids, image_tensors, image_sizes = zip(*batch)
        input_ids = torch.stack(input_ids, dim=0)
        image_tensors = torch.stack(image_tensors, dim=0)
        return input_ids, image_tensors, image_sizes


    # DataLoader
    def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=1):
        assert batch_size == 1, "batch_size must be 1"
        dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
        return data_loader


    eval_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'eval')
    question_file = os.path.join(eval_dir, 'textvqa/llava_textvqa_val_v051_ocr.jsonl')
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    image_folder = os.path.join(eval_dir, 'textvqa/train_images')
    answers_file = os.path.join(eval_dir, f'textvqa/answer_file_{model_name}.jsonl')
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, image_folder, tokenizer, model.image_processor, model.config)


    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    temperature = 0
    top_p = None
    num_beams = 1
    max_new_tokens = 128

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
    ans_file.close()

    print(f'''
    python -m llava.eval.eval_textvqa \
    --annotation-file {eval_dir}/textvqa/TextVQA_0.5.1_val.json \
    --result-file {answers_file}
    ''')

def train_with_hf_trainer():
    model_name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'

    parser = transformers.HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    print('local_rank', local_rank)
    if local_rank == 0:
        print(data_args)
        print(model_args)
        print(training_args)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.unk_token = "<|reserved_special_token_0|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.conv_version == 'plain':  # for pretrain
        conv_llava = Conversation(
            system="",
            roles=("", ""),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.PLAIN,
            sep="\n",
        )
    elif model_args.conv_version == 'llama_3': # for fine tune
        conv_llava = Conversation(
            system="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
            roles=("user", "assistant"),
            version="llama_v3",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.LLAMA_3,
            stop_token_ids=[128009],
            sep='<|start_header_id|>assistant<|end_header_id|>\n\n',
            sep2='<|start_header_id|>user<|end_header_id|>\n\n'
        )
    else:
        raise ValueError("wrong conv_version")

    model = DebugLlavaForCausalLM.from_pretrained(model_name_or_path, cache_dir=training_args.cache_dir, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    model.initialize_vision_modules(device=training_args.device, dtype=torch.bfloat16, pretrain_ckpt_path=model_args.pretrain_ckpt_path)
    model.to(training_args.device)

    training_args.conv_version = model_args.conv_version
    if model_args.conv_version == 'plain': 
        # self.model.requires_grad_(False)
        # self.lm_head.requires_grad_(False)
        # self.vision_tower.requires_grad_(False)
        # self.image_newline.requires_grad_(False)
        model.requires_grad_(False)
        for p in model.mm_projector.parameters():
            p.requires_grad = True
    else:
        # model.vision_tower.requires_grad_(False)
        lr_of_vit = training_args.mm_vision_tower_lr if training_args.mm_vision_tower_lr is not None else training_args.learning_rate
        lr_of_mlp = training_args.mm_projector_lr if training_args.mm_projector_lr is not None else training_args.learning_rate
        training_args.mm_projector_lr = lr_of_mlp
    
    if local_rank == 0:
        # Calculate total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('total_params: ', total_params)
        print('trainable_params: ', trainable_params)
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name, p.shape)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs=dict(use_reentrant=False)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    data_args.image_processor = model.image_processor
    data_args.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer, conversation=conv_llava, data_args=data_args)
    print('len train dataset', len(data_module['train_dataset']))

    model.config.use_cache = False
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    if data_args.image_aspect_ratio == 'anyres':
        base_size = model.vision_tower.config.image_size
        grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3]]
        # grids = [[1, 2], [2, 1], [2, 2]]
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints = [
            [g[0]*base_size, g[1]*base_size] for g in grids]
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr

    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    model.config.pretrain_ckpt_path = model_args.pretrain_ckpt_path
    # print(f'loading from pretrained checkpoint {model_args.pretrain_ckpt_path}')
    # if model_args.pretrain_ckpt_path is not None:
    #     unwrapped_model = trainer.accelerator.unwrap_model(model)
    #     load_sharded_checkpoint(unwrapped_model, model_args.pretrain_ckpt_path, strict=False)
    #     # load_sharded_checkpoint(trainer.model, model_args.pretrain_ckpt_path, strict=False)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print('loading from previous checkpoing')
        trainer.train(resume_from_checkpoint=True)
    else:
        print('training from scratch')
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if training_args.fsdp is not None and len(training_args.fsdp) > 0:
        trainer_save_model_safe_fsdp(trainer=trainer, output_dir=training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


def test_wds():



    tmp_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'])
    model_name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model_max_length = 8192
    data_args = DataArguments()
    data_args.data_path = [
        os.path.join(tmp_dir, 'train_json', 'llava_image_.json'),
        os.path.join(tmp_dir, 'train_json', 'llava_med_alignment_500k_cleaned.json')
    ]
    data_args.image_folder = tmp_dir
    model_args = ModelArguments()
    model_args.conv_version = 'llama_3'

    training_args = TrainingArguments(
        output_dir=os.path.join(tmp_dir, 'output'),
        cache_dir='/data/zhongz2/data/cache_dir' 
    )


    # parser = transformers.HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    # data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.unk_token = "<|reserved_special_token_0|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.conv_version == 'plain':  # for pretrain
        conv_llava = Conversation(
            system="",
            roles=("", ""),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.PLAIN,
            sep="\n",
        )
    elif model_args.conv_version == 'llama_3': # for fine tune
        conv_llava = Conversation(
            system="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
            roles=("user", "assistant"),
            version="llama_v3_v2",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.LLAMA_3,
            stop_token_ids=[128009],
            sep='<|start_header_id|>assistant<|end_header_id|>\n\n',
            sep2='<|start_header_id|>user<|end_header_id|>\n\n'
        )
    else:
        raise ValueError("wrong conv_version")


    vision_tower_name_or_ckpt = 'openai/clip-vit-large-patch14-336'
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name_or_ckpt)


    data_args.image_processor = image_processor # model.image_processor
    data_args.is_multimodal = True
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, conversation=conv_llava, data_args=data_args)

    save_dir = os.path.join(tmp_dir, 'data_{}'.format(model_args.conv_version))
    os.makedirs(save_dir, exist_ok=True)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    for i, item in enumerate(dataloader): 
        torch.save(item, os.path.join(save_dir, f'{i}.pt'))  # 1.3M per sample, too much disk space usage
        if i == 100:
            break

if __name__ == '__main__':
    train_with_hf_trainer()
    # train_with_deepspeed()
    