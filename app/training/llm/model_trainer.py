from datetime import datetime
from loguru import logger
import requests
import torch
import os

from fastapi.responses import JSONResponse
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
)

from datasets import load_dataset
from app.training.llm.std_writer import CustomStdErrWriter

from app.training.schemas.training import (
    FinetuningRequestSchema,
)
from app.training.services.training import TrainingService
from core.helpers.cache import *
from core.utils.file_util import *
from core.config import config


async def train_model(training_param: FinetuningRequestSchema):
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pre-trained model and tokenizer
    model_name = training_param.pm_name
    path = os.path.join(config.MODELS_DIR, model_name)

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
        callbacks=[HealthCheckCallback(model_name)],
    )

    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    logger.info(f"Training completed for {model_name} at {start_time}")

    # Send the progress via socket
    std_writer = CustomStdErrWriter(model_name)

    loss = ""

    try:
        Cache.set(TRAINING_CONTINUE, "True")
        result = trainer.train()
        loss = result["training_loss"]
    finally:
        # Restore stderr
        std_writer.close()

    end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # After training, save fine-tuned model, sessions, and parameters to the database
    ts_model_name = (
        f"{training_param.fm_name}_epoch_{training_param.epochs}_loss_{loss}"
    )
    ts_path = os.path.join(path, ts_model_name)

    logger.info(
        f"Training completed for {model_name} at {end_time}, result: {result}, ts_path: {ts_path}"
    )

    trainer.save_model(ts_path)

    await TrainingService().create_finetuning_model(
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
            logger.debug(e)
            raise e

    return _tokenize


class HealthCheckCallback(TrainerCallback):
    def __init__(self, repo_id):
        self.repo_id = repo_id

    def on_log(self, args, state, control, **kwargs):
        # Implement a simple check to the health endpoint.
        # If the endpoint is down, set `control.should_training_stop` to True.
        try:
            response = requests.get("http://0.0.0.0:8000/health")
            if response.status_code != 200:
                control.should_training_stop = "True"
                Cache.delete(f"{self.repo_id}_training")
                logger.debug(
                    f"Training stopped"
                )  # Change from logging.debug to logger.debug
        except Exception as e:  # Capture the exception for a detailed log.
            control.should_training_stop = True
            Cache.delete(f"{self.repo_id}_training")
            logger.debug(f"Training stopped due to error: {str(e)}")

        training_should_continue = Cache.get(TRAINING_CONTINUE)

        if not training_should_continue == "True":
            control.should_training_stop = True
            Cache.delete(f"{self.repo_id}_training")
            logger.debug(f"Training stopped due to stop_training command")
            return
