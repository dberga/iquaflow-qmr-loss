#FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04
#FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

# This prevents interactive region dialoge
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/nvidia/bin:$PATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN echo "**** Installing Python ****" && \
    add-apt-repository -y ppa:deadsnakes/ppa &&  \
    apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-distutils python3-apt 

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

RUN curl -O https://bootstrap.pypa.io/pip/3.6/get-pip.py
RUN python3.6 get-pip.py --isolated
RUN rm -rf /var/lib/apt/lists/* && \
    alias python3=python3.6

RUN apt-get update &&  \
    apt-get install -y \
    curl unzip wget git \
    ffmpeg libsm6 libxext6 libglib2.0-0 libgl1-mesa-glx

RUN echo 'alias pip=pip3.6' >> ~/.bashrc
RUN echo 'alias pip3=pip3.6' >> ~/.bashrc

RUN ln -sf /bin/python3.6 /bin/python
RUN ln -sf /bin/python3.6 /bin/python3

#RUN ln -s /usr/local/lib/python3.6/dist-packages/iq_tool_box/ /sisr/framework

RUN pip3 install pip --upgrade

WORKDIR /qmrloss
RUN pip3 install git+https://YOUR_GIT_TOKEN@github.com/satellogic/iquaflow.git

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install rasterio==1.2.6
RUN pip install kornia==0.6.4 --no-deps
#RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

#RUN apt-get install pandoc texlive-xetex texlive-fonts-recommended texlive-plain-generic # to latex-export pdf from jupyter
#ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["/bin/bash", "-c", "/bin/bash && tail -f /dev/null"]

