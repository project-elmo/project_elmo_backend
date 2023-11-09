import os
import shutil
from typing import List
from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from loguru import logger
from app.history.schemas.history import (
    TrainingSessionResponseSchema,
)
from app.inference.inference import answer_with_pdf, execute_inference
from app.inference.models.test import Test
from app.inference.schemas.inference import (
    GetTestListResponseSchema,
    MessageResponseSchema,
    MessageRequestSchema,
    TestResponseSchema,
)
from app.inference.services.inference import InferenceService
from app.training.models.finetuning_model import FinetuningModel
from app.training.models.training_session import TrainingSession
from app.training.schemas.training import DatasetResponseSchema
from app.user.schemas import ExceptionResponseSchema

from core.config import config
from core.helpers.cache import *
from core.utils.file_util import *

test_router = APIRouter()


@test_router.get(
    "/tests",
    response_model=List[GetTestListResponseSchema],
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def get_all_test():
    fm_models: List[FinetuningModel] = await InferenceService().get_all_fm_with_test()

    response_data = []

    for fm in fm_models:
        sessions = []
        tests = []

        training_sessions: List[TrainingSession] = fm.training_sessions

        for ts in training_sessions:
            sessions.append(
                TrainingSessionResponseSchema(
                    session_no=str(ts.session_no),
                    fm_no=fm.fm_no,
                    fm_name=fm.fm_name,
                    pm_no=ts.pm_no,
                    pm_name=ts.pm_name,
                    parent_session_no=str(ts.parent_session_no),
                    start_time=ts.start_time,
                    end_time=ts.end_time,
                    ts_model_name=ts.ts_model_name,
                )
            )

            test: Test = ts.tests
            if test:
                tests.append(
                    TestResponseSchema(
                        test_no=test.test_no,
                        session_no=test.session_no,
                        ts_model_name=ts.ts_model_name,
                        fm_no=test.fm_no,
                        fm_name=test.fm_name,
                    )
                )

        fm_data = GetTestListResponseSchema(
            fm_no=fm.fm_no, fm_name=fm.fm_name, list_sessions=sessions, list_test=tests
        )

        response_data.append(fm_data)

    return response_data


@test_router.post(
    "/create_test",
    response_model=TestResponseSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def create_test(session_no: int):
    """
    Create a test by session number if it does not exist; otherwise, return the existing test.
    """
    test = await InferenceService().create_test(session_no)
    return test


@test_router.post(
    "/create_message",
    response_model=List[MessageResponseSchema],
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def get_msg_resposne(msg_request: MessageRequestSchema):
    """
    Create message by user's input

    Returns:
        MessageResponseSchema: The result of the inference.
    """
    if msg_request.pdf_file_name == "":
        response = await execute_inference(msg_request)
    else:
        response = await answer_with_pdf(msg_request)

    results = await InferenceService().create_messages(msg_request, response)

    return results


@test_router.get(
    "/chat_history",
    response_model=List[MessageResponseSchema],
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def get_chat_history(test_no: int):
    response = await InferenceService().get_chat_history(test_no)
    return response


@test_router.post("/pdf_upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        datasets_path = config.PDF_DIR
        file_location = os.path.join(datasets_path, file.filename)

        logger.info(f"Saving file to {file_location}")

        with open(file_location, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("File saved successfully.")

        return {"filename": file.filename}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(detail=str(e), status_code=500)


@test_router.get("/download/{filename}")
async def download_pdf(filename: str):
    pdf_path = config.PDF_DIR
    file_path = os.path.join(pdf_path, filename)

    # Ensure file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path, filename=filename, media_type="application/octet-stream"
    )


@test_router.get("/get_pdf_files")
async def get_pdf_files():
    pdf_path = config.PDF_DIR

    # Check if the datasets directory exists
    if not os.path.isdir(pdf_path):
        raise HTTPException(status_code=404, detail="Datasets directory not found")

    extensions: list = ["pdf"]
    datasets = []

    for filename in os.listdir(pdf_path):
        if filename == ".gitkeep":
            continue

        file_path = os.path.join(pdf_path, filename)

        if os.path.isfile(file_path):
            _, file_extension = os.path.splitext(filename)
            if file_extension[1:] not in extensions:
                continue

            download_link = f"/download/{filename}"
            data = DatasetResponseSchema(
                file_path=file_path,
                size=os.path.getsize(file_path),
                filename=filename,
                extension=file_extension[1:],
                download_link=download_link,
            )
            datasets.append(data)

    return datasets
