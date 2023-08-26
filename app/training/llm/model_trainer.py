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

from app.training.schemas.training import FinetuningRequestSchema
from app.training.services.training import TrainingService
from core.helpers.cache import Cache
from core.config import config


async def train_model(training_param: FinetuningRequestSchema):
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pre-trained model and tokenizer
    model_name_lower = training_param.pm_name.lower()
    path = config.MODELS_DIR

    # TEMP
    return
