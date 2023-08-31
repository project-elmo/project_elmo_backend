from loguru import logger
import torch
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
from app.inference.schemas.inference import MessageRequestSchema
from app.inference.services.inference import InferenceService
from app.training.llm.model_util import (
    get_model_file_path,
    initialize_model,
    initialize_tokenizer,
)
from app.training.models.training_session import TrainingSession


def get_answer_with_context(
    question: str, context: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> str:
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].tolist()[0]

    answer_start_scores, answer_end_scores = model(**inputs)

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )

    return answer


def generate_answer(
    question: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> str:
    input_text = question
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = generated_answer.replace(input_text, "").strip()
    logger.info(answer)
    return answer


def generate_text(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 50,
) -> str:
    """
    For text generation
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


async def execute_inference(request_schema: MessageRequestSchema) -> str:
    session_model: TrainingSession = await InferenceService().get_session_by_session_no(
        request_schema.session_no
    )
    model_path = get_model_file_path(
        session_model.pm_name, session_model.fm_name, session_model.uuid
    )
    model = initialize_model(model_path)
    tokenizer = initialize_tokenizer(session_model.pm_name)
    prompt = request_schema.prompt

    response = ""
    if request_schema.task == 0:
        response = generate_answer(prompt, model, tokenizer)
    elif request_schema.task == 2:
        response = generate_text(prompt, model, tokenizer)

    return response
