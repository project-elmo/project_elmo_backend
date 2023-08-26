FROM jharinn/elmo:gpu
WORKDIR /home

ENV ENV=dev

ENV DOCKERIZE_VERSION v0.6.1
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/jwilder/dockerize/releases/download/$DOCKERIZE_VERSION/dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && tar -C /usr/local/bin -xzvf dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && rm dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz

RUN pip install "poetry==1.6.0" \
    && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /home/
RUN poetry install --no-dev

COPY . /home

RUN chmod +x /home/docker/api/startup.sh
ENTRYPOINT ["/home/docker/api/startup.sh"]
