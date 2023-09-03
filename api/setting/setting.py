from fastapi import APIRouter

from app.setting.schemas.setting import SettingSchema
from app.setting.services.setting import SettingService
from app.user.schemas import (
    ExceptionResponseSchema,
)
import torch

setting_router = APIRouter()


@setting_router.get(
    "/is_gpu_available",
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def check_gpu():
    is_gpu = torch.cuda.is_available()
    return {"is_gpu_available": is_gpu}


@setting_router.post(
    "/create_setting",
    response_model=SettingSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def create_setting(setting: SettingSchema):
    return await SettingService().create_setting(setting)


@setting_router.post(
    "/get_setting",
    response_model=SettingSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def get_setting():
    return await SettingService().get_setting()


@setting_router.post(
    "/update_setting",
    response_model=SettingSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def update_setting(setting: SettingSchema):
    return await SettingService().update_setting(setting)
