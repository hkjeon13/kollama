FROM huggingface/transformers-pytorch-gpu

# Install Python 3.10 and essential tools
RUN apt update -y && apt-get update -y \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get install -y python3.10 python3.10-distutils python3.10-dev \
                          git git-lfs curl wget vim build-essential libopenmpi-dev

# Set Python 3.10 as the default python and python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Install PyTorch and additional libraries
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 \
    && pip3 install transformers datasets accelerate evaluate deepspeed slacker seaborn scikit-learn wandb mpi4py \
    && pip3 install fsspec==2023.6.0 datasets==2.12.0 huggingface-hub==0.17.0
