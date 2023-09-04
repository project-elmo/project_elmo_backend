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
from app.setting.services.setting import SettingService


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
    request_schema: MessageRequestSchema,
) -> str:
    input_text = question
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(
        input_ids,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        max_length=request_schema.max_length,
        temperature=request_schema.temperature,
        top_k=request_schema.top_k,
        top_p=request_schema.top_p,
        repetition_penalty=request_schema.repetition_penalty,
        no_repeat_ngram_size=request_schema.no_repeat_ngram_size,
    )

    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = generated_answer.replace(input_text, "").strip()
    logger.info(answer)
    return answer


async def execute_inference(request_schema: MessageRequestSchema) -> str:
    session_model: TrainingSession = await InferenceService().get_session_by_test_no(
        request_schema.test_no
    )
    model_path = get_model_file_path(
        session_model.pm_name, session_model.fm_name, session_model.uuid
    )
    tokenizer = initialize_tokenizer(session_model.pm_name)
    model = initialize_model(model_path)

    if SettingService().get_is_gpu == "true":
        model.to("cuda:0")

    logger.info(f"model_path:{model_path}, model:{model.config}")
    prompt = request_schema.msg

    response = ""
    if request_schema.task == 0 or request_schema.task == 2:
        response = generate_answer(prompt, model, tokenizer, request_schema)

    return response
