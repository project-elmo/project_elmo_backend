#!/bin/bash
dockerize -wait tcp://db:3306 -timeout 20s
alembic upgrade head && gunicorn --bind 0.0.0.0:8000 --workers 8 --threads 8 --timeout 3000 -k uvicorn.workers.UvicornWorker app.server:app