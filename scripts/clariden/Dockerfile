# https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/pyg/-
# FROM nvcr.io/nvidia/pyg:26.09-py3 as base

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

# https://developer.nvidia.com/cusparselt-downloads?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=Ubuntu&target_version=24.04&target_type=deb_network
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \
	&& dpkg -i cuda-keyring_1.1-1_all.deb \
	&& rm cuda-keyring_1.1-1_all.deb \
	&& apt-get update && apt-get install -y \
	cusparselt-cuda-13 \
	&& apt-get clean && rm -rf /var/lib/apt/lists/*
