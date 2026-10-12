# https://docs.cscs.ch/software/alps-extended-images/
FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.02-py3-alps6 as base
# Ubuntu 24.04
# PyTorch 2.11.0a0+eb65b36914

ENV DEBIAN_FRONTEND=noninteractive \
	PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=off \
	PIP_DISABLE_PIP_VERSION_CHECK=on \
	PIP_DEFAULT_TIMEOUT=100

RUN apt-get update && apt-get install -y \
	curl \
	python3.12-dev \
	python3.12-venv \
	libpq-dev \
	build-essential \
	wget
