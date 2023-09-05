#!/bin/bash
dockerize -wait tcp://db:3306 -timeout 20s
alembic upgrade head && gunicorn --bind 0.0.0.0:80 --workers 4 --threads 8 --timeout 10000 -k uvicorn.workers.UvicornWorker app.server:app --reload