import os

from pydantic_settings import BaseSettings
from typing import Optional


class Config(BaseSettings):
    ENV: str = "dev"
    DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 80
    WRITER_DB_URL: str = f"mysql+aiomysql://root:fastapi@db:3306/fastapi?charset=utf8"
    READER_DB_URL: str = (
        f"mysql+aiomysql://root:fastapi@localhost:3306/fastapi?charset=utf8"
    )
    JWT_SECRET_KEY: str = "fastapi"
    JWT_ALGORITHM: str = "HS256"
    SENTRY_SDN: Optional[str] = None
    CELERY_BROKER_URL: str = "amqp://user:bitnami@localhost:5672/"
    CELERY_BACKEND_URL: str = "redis://:password123@localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    DL_DIR: str = os.path.join(os.environ["HOME"], ".cache", "huggingface")
    RESULT_DIR: str = os.path.join(DL_DIR, "elmo", "result")
    DATASET_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"
    )
    PDF_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pdf"
    )


class DevelopmentConfig(Config):
    WRITER_DB_URL: str = f"mysql+aiomysql://root:fastapi@db:3306/fastapi?charset=utf8"
    READER_DB_URL: str = f"mysql+aiomysql://root:fastapi@db:3306/fastapi?charset=utf8"
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379


class LocalConfig(Config):
    WRITER_DB_URL: str = (
        f"mysql+aiomysql://root:fastapi@localhost:3306/fastapi?charset=utf8"
    )
    READER_DB_URL: str = (
        f"mysql+aiomysql://root:fastapi@localhost:3306/fastapi?charset=utf8"
    )


class ProductionConfig(Config):
    DEBUG: bool = False
    WRITER_DB_URL: str = (
        f"mysql+aiomysql://root:fastapi@localhost:3306/prod?charset=utf8"
    )
    READER_DB_URL: str = (
        f"mysql+aiomysql://root:fastapi@localhost:3306/prod?charset=utf8"
    )


def get_config():
    env = os.getenv("ENV", "local")
    config_type = {
        "dev": DevelopmentConfig(),
        "local": LocalConfig(),
        "prod": ProductionConfig(),
    }
    return config_type[env]


config: Config = get_config()
