from fastapi import APIRouter

from app.setting.schemas.setting import SettingSchema
from app.setting.services.setting import SettingService
from app.user.schemas import (
    ExceptionResponseSchema,
)

setting_router = APIRouter()


@setting_router.post(
    "/create_setting",
    response_model=SettingSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def create_setting(setting: SettingSchema):
    return await SettingService().create_setting(setting)


@setting_router.get(
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
