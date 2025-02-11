FROM nvcr.io/nvidia/pytorch:24.10-py3 as base

ENV DEBIAN_FRONTEND=noninteractive \
	PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=off \
	PIP_DISABLE_PIP_VERSION_CHECK=on \
	PIP_DEFAULT_TIMEOUT=100 \
	POETRY_HOME="/opt/poetry" \
	POETRY_VIRTUALENVS_IN_PROJECT=true \
	POETRY_VIRTUALENVS_OPTIONS_SYSTEM_SITE_PACKAGES=true \
	POETRY_NO_INTERACTION=1 \
	PYSETUP_PATH="/opt/pysetup" \
	VENV_PATH="/opt/pysetup/.venv" \
	POETRY_VERSION=1.8.4

RUN apt-get update \
	&& apt-get install -y \
	curl \
	python3-dev \
	libpq-dev \
	build-essential \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

FROM base as final-stage

WORKDIR $PYSETUP_PATH

COPY poetry.lock ./

COPY pyproject.toml ./

RUN poetry install --no-root --only main
