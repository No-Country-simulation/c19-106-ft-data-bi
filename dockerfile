FROM python:3.11.9-slim-bullseye

WORKDIR /app


ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .