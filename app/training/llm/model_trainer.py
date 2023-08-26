import torch
import os
import logging

from fastapi.responses import JSONResponse
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    BertTokenizer,
    BertLMHeadModel,
    RobertaTokenizer,
    RobertaForCausalLM,
    AlbertTokenizer,
    AlbertModel,
    ElectraTokenizer,
    ElectraForCausalLM,
    AutoTokenizer,
    LlamaModel,
    AutoModel,
    Trainer,
    TrainingArguments,
)

from app.training.models.training_parameter import TrainingParameter
from app.training.services.training import TrainingService
from core.helpers.cache import Cache
from core.config import config


async def train_model(training_param: TrainingParameter):
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pre-trained model and tokenizer
    model_name_lower = training_param.model_name.lower()
    path = config.MODELS_DIR

    if "gpt2" in model_name_lower:
        tokenizer = GPT2Tokenizer.from_pretrained(training_param.model_name)
        model = GPT2LMHeadModel.from_pretrained(
            training_param.model_name, local_files_only=True, cache_dir=path
        )
    elif "bert" in model_name_lower:
        tokenizer = BertTokenizer.from_pretrained(training_param.model_name)
        model = BertLMHeadModel.from_pretrained(
            training_param.model_name, local_files_only=True, cache_dir=path
        )
    elif "roberta" in model_name_lower:
        tokenizer = RobertaTokenizer.from_pretrained(training_param.model_name)
        model = RobertaForCausalLM.from_pretrained(
            training_param.model_name, local_files_only=True, cache_dir=path
        )
    elif "albert" in model_name_lower:
        tokenizer = AlbertTokenizer.from_pretrained(training_param.model_name)
        model = AlbertModel.from_pretrained(
            training_param.model_name, local_files_only=True, cache_dir=path
        )
    elif "electra" in model_name_lower:
        tokenizer = ElectraTokenizer.from_pretrained(training_param.model_name)
        model = ElectraForCausalLM.from_pretrained(
            training_param.model_name, local_files_only=True, cache_dir=path
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(training_param.model_name)
        model = AutoModel.from_pretrained(
            training_param.model_name, local_files_only=True, cache_dir=path
        )
