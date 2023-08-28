from datetime import datetime
from loguru import logger
import torch
import os
import logging

from fastapi.responses import JSONResponse
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
)
from datasets import load_dataset

from app.training.schemas.training import (
    FinetuningRequestSchema,
)
from app.training.services.training import TrainingService
from core.helpers.cache import Cache
from core.config import config
from core.utils.file_util import *


def train_model(training_param: FinetuningRequestSchema):
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pre-trained model and tokenizer
    model_name = training_param.pm_name
    path = config.MODELS_DIR

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name, local_files_only=True)
    else:
        model = AutoModel.from_pretrained(model_name, local_files_only=True)

    # Load the dataset
    dataset_path = training_param.dataset
    loaders = {"json": "json", "csv": "csv"}

    extension = get_extension_from_path(dataset_path)

    logger.debug(f"dataset_path: {dataset_path}, extension: {extension}")
    dataset = load_dataset(
        loaders.get(extension, "default_value"), data_files=dataset_path
    )

    logger.debug(f"{dataset}")

    tokenized_datasets = dataset.map(tokenize_qa(tokenizer), batched=True)

    # Set up the training arguments
    if training_param.save_total_limits == -1:
        training_args = TrainingArguments(
            output_dir=config.RESULT_DIR,
            num_train_epochs=training_param.epochs,
            per_device_train_batch_size=training_param.batch_size,
            evaluation_strategy=training_param.evaluation_strategy,
            learning_rate=training_param.learning_rate,
            weight_decay=training_param.weight_decay,
            logging_strategy=training_param.logging_strategy,
            logging_steps=training_param.save_steps,
            eval_steps=training_param.eval_steps,
            save_strategy=training_param.save_strategy,
        )
    else:
        training_args = TrainingArguments(
            output_dir=config.RESULT_DIR,
            num_train_epochs=training_param.epochs,
            per_device_train_batch_size=training_param.batch_size,
            evaluation_strategy=training_param.evaluation_strategy,
            learning_rate=training_param.learning_rate,
            weight_decay=training_param.weight_decay,
            save_total_limit=training_param.save_total_limits,  # set save_total_limits
            logging_strategy=training_param.logging_strategy,
            logging_steps=training_param.save_steps,
            eval_steps=training_param.eval_steps,
            save_strategy=training_param.save_strategy,
        )

    # Start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    logging.info(f"Training completed for {model_name} at {start_time}")

    # training
    trainer.train()
    end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    trainer.save_model(os.path.join(path, ts_model_name))

    # compute train results
    logging.info(f"Training completed for {model_name} at {end_time}")

    # After training, save fine-tuned model, sessions, and parameters to the database
    ts_model_name = f"{training_param.fm_name}_{training_param.epochs}"
    TrainingService().create_finetuning_model(
        training_param=training_param,
        start_time=start_time,
        end_time=end_time,
        ts_model_name=ts_model_name,
    )


def tokenize_qa(tokenizer: PreTrainedTokenizer):
    def _tokenize(batch):
        try:
            encoding = tokenizer(
                batch["question"],
                batch["answer"],
                truncation=True,
                padding="max_length",
                max_length=512,
            )
            encoding["labels"] = encoding["input_ids"].copy()
            return encoding
        except KeyError as e:
            logging.debug(e)
            raise e

    return _tokenize
