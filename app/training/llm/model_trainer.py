from datetime import datetime
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
    model_name = training_param.pm_name
    path = config.MODELS_DIR

    if "gpt2" in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(
            model_name, local_files_only=True, cache_dir=path
        )
    elif "bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertLMHeadModel.from_pretrained(
            model_name, local_files_only=True, cache_dir=path
        )
    elif "roberta" in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForCausalLM.from_pretrained(
            model_name, local_files_only=True, cache_dir=path
        )
    elif "albert" in model_name:
        tokenizer = AlbertTokenizer.from_pretrained(model_name)
        model = AlbertModel.from_pretrained(
            model_name, local_files_only=True, cache_dir=path
        )
    elif "electra" in model_name:
        tokenizer = ElectraTokenizer.from_pretrained(model_name)
        model = ElectraForCausalLM.from_pretrained(
            model_name, local_files_only=True, cache_dir=path
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name, local_files_only=True, cache_dir=path
        )

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=config.RESULT_DIR,
        num_train_epochs=training_param.epochs,
        per_device_train_batch_size=training_param.batch_size,
        evaluation_strategy=training_param.evaluation_strategy,
        learning_rate=training_param.learning_rate,
        save_total_limit=training_param.save_total_limits,
        logging_strategy=training_param.logging_strategy,
        logging_steps=training_param.save_steps,  # Assuming this is the correct mapping
        eval_steps=training_param.eval_steps,
        save_strategy=training_param.save_strategy,
        disable_tqdm=False,
        report_to=None,
        return_dict=True,
        device=device,  # Set device based on CUDA availability
    )

    # Start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenizer(training_param.dataset),
    )

    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    # training
    trainer.train()
    end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # compute train results
    logging.info(f"Training completed for {model_name}")

    # After training, save fine-tuned model, sessions, and parameters to the database
    ts_model_name = f"{training_param.fm_name}_{training_param.epochs}"
    TrainingService().create_finetuning_model(
        training_param=training_param,
        start_time=start_time,
        end_time=end_time,
        ts_model_name=ts_model_name,
    )
