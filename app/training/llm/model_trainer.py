from typing import Any, Callable, NamedTuple, Dict, Union, Tuple
from datetime import datetime
from loguru import logger
import uuid
import requests
import torch
import os
import csv
import json

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

from datasets import load_dataset, Dataset
from app.setting.services.setting import SettingService
from app.training.download.progress import set_result
from app.training.llm.model_util import (
    get_model_file_path,
    initialize_model,
    initialize_tokenizer,
)
from app.training.llm.std_writer import CustomStdErrWriter
from app.training.models import FinetuningModel, TrainingSession

from app.training.schemas.training import (
    FinetuningRequestSchema,
    TrainingSessionRequestSchema,
)
from app.training.services.training import TrainingService
from core.helpers.cache import *
from core.utils.file_util import *
from core.config import config


async def train_model(
    training_param: Union[FinetuningRequestSchema, TrainingSessionRequestSchema],
    pm_name: str,
    fm_name: str,
    initial_training: bool,
) -> Union[FinetuningModel, TrainingSession]:
    # Check if CUDA is available
    device = True if await get_setting_device() == "false" else False

    # Load the pre-trained model and tokenizer
    tokenizer = initialize_tokenizer(pm_name)
    model = initialize_model(pm_name)

    # Load the dataset
    tokenized_datasets = await load_and_tokenize_dataset(
        training_param.dataset, tokenizer, training_param
    )

    # Set Trainer
    training_args = get_training_args(training_param, device)
    trainer: Trainer = setup_training(
        model, training_args, tokenized_datasets, training_param, pm_name
    )

    # Start training
    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    logger.info(f"Training started for {pm_name} at {start_time}")

    # Send the progress via socket
    result = await send_progress(
        start_training, trainer, training_param, initial_training, pm_name
    )

    end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    training_param.ts_model_name = training_param.ts_model_name or get_model_name(
        training_param, result, initial_training
    )
    uuid = get_uuid()
    # home/.cache/hugginggace/elmo/result/{model_name}/{fm_name}/{uuid}
    save_path = get_model_file_path(pm_name, fm_name, uuid)
    logger.info(
        f"Training completed for {pm_name} at {end_time}, result: {result}, ts_path: {save_path}"
    )

    # Save Model
    trainer.save_model(save_path)

    # Insert into DB
    if initial_training:
        finetuning_model: FinetuningModel = (
            await TrainingService().create_finetuning_model(
                training_param=training_param,
                start_time=start_time,
                end_time=end_time,
                uuid=uuid,
            )
        )

        return finetuning_model
    else:
        # Set paraent_session_no
        if not training_param.parent_session_no:
            training_param.parent_session_no = "0"
        session: TrainingSession = await TrainingService().create_training_session(
            training_param=training_param,
            start_time=start_time,
            end_time=end_time,
            uuid=uuid,
        )

        return session


def tokenize_qa(
    tokenizer: PreTrainedTokenizer,
    training_param: Union[FinetuningRequestSchema, TrainingSessionRequestSchema],
):
    def _tokenize(batch):
        try:
            encoding = tokenizer(
                batch[f"{training_param.keys_to_use[0]}"],
                batch[f"{training_param.keys_to_use[1]}"],
                truncation=True,
                padding="max_length",
                max_length=training_param.max_length,
            )
            encoding["labels"] = encoding["input_ids"].copy()
            return encoding
        except KeyError as e:
            logger.debug(e)
            raise e

    return _tokenize


async def send_progress(
    func: Callable[..., NamedTuple], *args: Any, **kwargs: Dict[str, Any]
) -> NamedTuple:
    """
    Sends progress metrics of a given function using a socket.
    """
    model_name = kwargs["pm_name"]
    logger.info(f"model_name: {model_name}")

    # Send the progress via socket
    std_writer = CustomStdErrWriter(model_name)

    try:
        result = await func(*args)

        result_msg = {"task": "task_result", "model_name": model_name}

        metrics: Dict[str, Union[str, float, int]] = result[2]
        result_msg.update(metrics)
        metrics_json = json.dumps(result_msg)

        logger.info(f"metrics_json: {metrics_json}")
        set_result(model_name, metrics_json)

    finally:
        # Restore stderr
        std_writer.close()

    return result


async def get_setting_device() -> str:
    is_gpu = await SettingService().get_is_gpu()

    return is_gpu


async def load_and_tokenize_dataset(
    dataset_path: str,
    tokenizer,
    training_param: Union[FinetuningRequestSchema, TrainingSessionRequestSchema],
) -> Dataset:
    """
    Loads the dataset from the given path and tokenizes it.

    Args:
        dataset_path (str): Path to the dataset file.
        tokenizer: Tokenizer to be used for tokenization.
        task (int): Task identifier.

    Returns:
        Dataset: Tokenized dataset.
    """
    loaders = {"json": "json", "csv": "csv"}
    extension = get_extension_from_path(dataset_path)
    dataset = load_dataset(
        loaders.get(extension, "default_value"), data_files=dataset_path
    )

    if training_param.task == 0:
        return dataset.map(tokenize_qa(tokenizer, training_param), batched=True)
    elif training_param.task == 1:
        return dataset.map(tokenize_qa(tokenizer, training_param), batched=True)
    elif training_param.task == 2:
        return dataset.map(tokenize_qa(tokenizer, training_param), batched=True)
    else:
        return dataset.map(tokenize_qa(tokenizer, training_param), batched=True)


def get_training_args(
    training_param: FinetuningRequestSchema, device: bool
) -> TrainingArguments:
    common_args = {
        "output_dir": config.RESULT_DIR,
        "num_train_epochs": training_param.epochs,
        "per_device_train_batch_size": training_param.batch_size,
        "evaluation_strategy": training_param.evaluation_strategy,
        "learning_rate": training_param.learning_rate,
        "weight_decay": training_param.weight_decay,
        "logging_strategy": training_param.logging_strategy,
        "logging_steps": training_param.save_steps,
        "eval_steps": training_param.eval_steps,
        "save_strategy": training_param.save_strategy,
        "use_cpu": not device,
    }

    if training_param.save_total_limits != -1:
        common_args["save_total_limit"] = training_param.save_total_limits

    return TrainingArguments(**common_args)


def setup_training(
    model: PreTrainedModel,
    training_args: TrainingArguments,
    tokenized_datasets: Dataset,
    training_param: Union[FinetuningRequestSchema, TrainingSessionRequestSchema],
    pm_name: str,
) -> Trainer:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        callbacks=[HealthCheckCallback(pm_name)],
    )

    return trainer


async def start_training(
    trainer: Trainer,
    training_param: Union[FinetuningRequestSchema, TrainingSessionRequestSchema],
    initial_training: bool,
    pm_name: str,
) -> NamedTuple:
    if initial_training:
        result: NamedTuple = trainer.train()
    else:
        uuid = await TrainingService().get_uuid_by_session_no(
            training_param.parent_session_no
        )
        resume_from_checkpoint = get_model_file_path(
            pm_name, training_param.fm_name, uuid
        )
        result: NamedTuple = trainer.train(os.path.join(resume_from_checkpoint))
    return result


def get_model_name(
    training_param: Union[FinetuningRequestSchema, TrainingSessionRequestSchema],
    result: NamedTuple,
    initial_training: bool,
) -> str:
    loss = result[1]
    base_name = f"{training_param.fm_name}_epoch_{training_param.epochs}_loss_{loss}"

    if initial_training:
        return f"{base_name}_0"
    else:
        return f"{base_name}_{training_param.parent_session_no}"


def get_uuid():
    uuid_full = uuid.uuid4()
    return str(uuid_full).split("-")[0] + "-" + str(uuid_full).split("-")[1]


class HealthCheckCallback(TrainerCallback):
    def __init__(self, repo_id):
        self.repo_id = repo_id

    def on_log(self, args, state, control, **kwargs):
        # Implement a simple check to the health endpoint.
        # If the endpoint is down, set `control.should_training_stop` to True.
        try:
            response = requests.get("http://0.0.0.0:80/health")
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


def extract_columns_from_csv(file_path: str) -> list:
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames


def extract_keys_from_json(file_path: str) -> list:
    with open(file_path, "r") as f:
        data = json.load(f)
        return data[0].keys() if isinstance(data, list) and data else []
