FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

#UTC ?
ENV TZ=Europe/Paris 
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y --no-install-recommends
RUN apt-get update
RUN apt-get install -y python3-pip python3-dev \
  && apt-get install -y libgeoip-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python

RUN apt-get install -y libgdal-dev
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get update && \
    apt-get --no-install-recommends -y install curl gdal-bin libglib2.0-0 libgl1-mesa-glx zip ffmpeg && \
    rm -rf /var/lib/apt /var/cache/apt

# installing 
RUN python -m pip install --upgrade pip 
RUN python -m pip install --use-feature=2020-resolver rasterio==1.0.13
RUN python -m pip install opencv-python==4.5.1.48
RUN python -m pip install tqdm==4.57.0
RUN python -m pip install kornia==0.4.1
RUN python -m pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

USER root

COPY . /MSRN_EXAMPLE
WORKDIR /MSRN_EXAMPLE

ENTRYPOINT ["python", "main.py"]