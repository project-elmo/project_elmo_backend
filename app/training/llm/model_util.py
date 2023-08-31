import os
from typing import Tuple
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
)

from core.config import config


def initialize_model_and_tokenizer(model_name_or_path: str) -> Tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if model_name_or_path == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(
            model_name_or_path, local_files_only=True
        )
    else:
        model = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)
    return model, tokenizer


def get_model_file_path(pm_name: str, fm_name: str, uuid: str) -> str:
    fm_name_path = os.path.join(config.RESULT_DIR, pm_name, fm_name)
    return os.path.join(fm_name_path, uuid)
