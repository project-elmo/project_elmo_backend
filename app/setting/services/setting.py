import torch
from loguru import logger
from sqlalchemy import select, update
from app.setting.models.setting import ElmoSetting
from app.setting.schemas.setting import SettingSchema
from core.helpers.cache.cache_keys import *
from core.helpers.cache import Cache

from core.db import session


class SettingService:
    async def get_setting(self) -> ElmoSetting:
        query = select(ElmoSetting).filter(ElmoSetting.set_no == 1)
        result = await session.execute(query)
        setting: ElmoSetting = result.scalar()
        setting.is_gpu = torch.cuda.is_available()
        logger.info(f"setting.is_gpu: {setting.is_gpu}")

        return setting

    async def create_setting(
        self,
        setting: SettingSchema = SettingSchema(),
    ) -> ElmoSetting:
        try:
            prev_setting = await self.get_setting()

            if prev_setting:
                prev_stting = await self.update_setting(setting)
                return prev_stting

            elmo_setting = ElmoSetting(
                model_path=setting.model_path,
                result_path=setting.result_path,
                is_gpu=torch.cuda.is_available(),            
            )

            session.add(elmo_setting)
            await session.commit()
            await session.refresh(elmo_setting)

            return elmo_setting

        except Exception as e:
            await session.rollback()
            logger.error(f"Error while inserting: {e}")
            raise e

    async def update_setting(self, setting: SettingSchema) -> ElmoSetting:
        try:
            query = (
                update(ElmoSetting)
                .where(ElmoSetting.set_no == 1)  # TODO: fix
                .values(
                    model_path=setting.model_path,
                    result_path=setting.result_path,
                    is_gpu=setting.is_gpu_use,
                )
            )
            await session.execute(query)
            await session.commit()

            setting = await self.get_setting()
            return setting

        except Exception as e:
            await session.rollback()
            logger.error(f"Error while updating: {e}")
            raise e

    async def get_is_gpu(self):
        is_gpu = torch.cuda.is_available()

        if is_gpu:
            return is_gpu
        else:
            setting: ElmoSetting = await self.get_setting()

            if not setting:
                setting = await self.create_setting()

            self.set_is_gpu(str(setting.is_gpu))
            return str(setting.is_gpu)

    def set_is_gpu(self, is_gpu: str):
        Cache.set(IS_GPU, str(is_gpu))
