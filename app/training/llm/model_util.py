import os
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from loguru import logger
from core.config import config


def initialize_model(model_name_or_path: str) -> PreTrainedModel:
    if "gpt2" == model_name_or_path.lower():
        model = GPT2LMHeadModel.from_pretrained(
            model_name_or_path, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, local_files_only=True
        )
    return model


def initialize_tokenizer(model_name: str) -> PreTrainedTokenizer:
    if "gpt2" == model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    elif "skt/kogpt2-base-v2" == model_name.lower():
        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token='</s>', eos_token='</s>', unk_token='<unk>',
            pad_token='<pad>', mask_token='<mask>')
    else:
       tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer


def get_model_file_path(pm_name: str, fm_name: str, uuid: str) -> str:
    fm_name_path = os.path.join(config.RESULT_DIR, pm_name, fm_name)
    return os.path.join(fm_name_path, uuid)
