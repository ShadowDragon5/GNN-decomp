FROM nvcr.io/nvidia/pytorch:24.10-py3 as base

ENV DEBIAN_FRONTEND=noninteractive \
	PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=off \
	PIP_DISABLE_PIP_VERSION_CHECK=on \
	PIP_DEFAULT_TIMEOUT=100

RUN apt-get update && apt-get install -y \
	curl \
	python3-dev \
	python3.10-venv \
	libpq-dev \
	build-essential \
	wget \
	&& wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb \
	&& dpkg -i cuda-keyring_1.1-1_all.deb \
	&& rm cuda-keyring_1.1-1_all.deb \
	&& apt-get update && apt-get install -y \
	libcusparselt0 \
	libcusparselt-dev \
	&& apt-get clean && rm -rf /var/lib/apt/lists/*
