from fastapi import APIRouter

from app.setting.schemas.setting import SettingSchema
from app.setting.services.setting import SettingService
from app.user.schemas import (
    ExceptionResponseSchema,
)
from app.user.schemas.user import CreateUserRequestSchema
from app.user.services.user import UserService

setting_router = APIRouter()


@setting_router.get(
    "/get_setting",
    response_model=SettingSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def get_setting():
    # insert initial data
    await UserService().insert_initial_data()
    return await SettingService().get_setting()


@setting_router.post(
    "/update_setting",
    response_model=SettingSchema,
    responses={"400": {"model": ExceptionResponseSchema}},
)
async def update_setting(setting: SettingSchema):
    return await SettingService().update_setting(setting)
