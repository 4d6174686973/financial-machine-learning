FROM python:3.9.12-slim-bullseye

RUN pip install --upgrade pip

RUN apt-get update && apt-get install --no-install-recommends  -y \
    git \
    openssh-client\
    && rm -rf /var/lib/apt/lists/* 

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN useradd -ms /bin/bash vscode
USER vscode
WORKDIR /workspace