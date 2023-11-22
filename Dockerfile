FROM huggingface/transformers-pytorch-cpu
RUN apt update -y 
RUN apt-get update -y  
RUN apt-get install git git-lfs curl wget software-properties-common vim -y  
RUN apt-get install build-essential -y  
RUN add-apt-repository ppa:deadsnakes/ppa -y  
RUN apt install python3.10 -y  
RUN apt install python3.10-distutils -y  
RUN apt install python3.10-dev -y  
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1  
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1  
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10  -y  
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -y  
RUN pip3 install transformers datasets accelerate evaluate deepspeed slacker seaborn scikit-learn -y  
RUN pip3 install fsspec==2023.6.0 datasets==2.12.0 huggingface-hub==0.17.0 -y
