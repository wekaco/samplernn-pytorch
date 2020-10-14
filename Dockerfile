FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu16.04

# Export CUDA env variables
# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-10.2/lib64"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda-10.2/lib64"
ENV CUDA_HOME="/usr/local/cuda"
# ENV PATH="/usr/local/cuda/bin:/usr/local/cuda-10.2/bin:$PATH"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    sox \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -L -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda update -n base -c defaults conda \
 && /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch \
  && conda clean -ya

# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
   pip install --no-cache-dir -r requirements.txt

WORKDIR /app

RUN mkdir -p datasets && mkdir -p results

COPY LICENSE /app/LICENSE
COPY README.md /app/README.md

COPY *.py /app/
COPY trainer /app/trainer
COPY gen /app/gen

COPY *.yaml /app/
