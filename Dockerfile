FROM huggingface/transformers-pytorch-cpu
RUN apt-get install git git-lfs curl wget -y 