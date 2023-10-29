import os
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
    pipeline,
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
from core.config import config

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA


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
        do_sample=True,
        top_k=request_schema.top_k,
        top_p=request_schema.top_p,
        repetition_penalty=request_schema.repetition_penalty,
        no_repeat_ngram_size=request_schema.no_repeat_ngram_size,
    )
    generated_answer = tokenizer.decode(
        output[0][len(input_ids[0]) :], skip_special_tokens=True
    )
    logger.info(tokenizer.decode(output[0]))
    logger.info(generated_answer)

    return generated_answer


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
    if request_schema.task == 0 or request_schema.task == 1:
        response = generate_answer(prompt, model, tokenizer, request_schema)

    return response


# TODO
def answer_with_web(url: str):
    loader = WebBaseLoader(url)
    data = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(data)

    # VectorDB
    embeddings_model = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(documents=splits, embedding=embeddings_model)


async def answer_with_pdf(request_schema: MessageRequestSchema):
    # init model
    session_model: TrainingSession = await InferenceService().get_session_by_test_no(
        request_schema.test_no
    )
    model_path = get_model_file_path(
        session_model.pm_name, session_model.fm_name, session_model.uuid
    )
    model = initialize_model(model_path)
    tokenizer = initialize_tokenizer(session_model.pm_name)

    logger.info(f"making a pipeline... model_path:{model_path}, model:{model.config}")
    # max_length has typically been deprecated for max_new_tokens
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=request_schema.max_length,
        model_kwargs={
            "max_length" : request_schema.max_length,
            "temperature": request_schema.temperature,
            "top_k": request_schema.top_k,
            "top_p": request_schema.top_p,
            "repetition_penalty": request_schema.repetition_penalty,
            "no_repeat_ngram_size": request_schema.no_repeat_ngram_size,
        },
    )

    if session_model.pm_name == "gpt2":
        pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

    hf_llm = HuggingFacePipeline(pipeline=pipe)

    pdf_dir_path = config.PDF_DIR
    path = os.path.join(pdf_dir_path, request_schema.pdf_file_name)
    loader = PyPDFLoader(path)

    # split the pdf file
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        length_function=len,
        is_separator_regex=False,
    )

    logger.info(f"split...")
    texts = text_splitter.split_documents(pages)

    logger.info(f"texts...{texts}")

    # embedding
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

    if request_schema.lang == "ko":
        embedding_model_name = "jhgan/ko-sroberta-multitask"
    embeddings_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
    # model_kwargs={'device': 'cuda'},
    )

    logger.info(f"embeddings_model...{embeddings_model}")

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    qa_chain = RetrievalQA.from_chain_type(
        llm=hf_llm,
        retriever=db.as_retriever(),
    )

    logger.info(f"qa_chain...{qa_chain}")

    result = qa_chain( 
        {
            "query": f"{request_schema.msg}",
            "token_max": model.config.max_length,
        },
        return_only_outputs=True,
    )

    logger.info(f"result... {result}")
    return result['result']
