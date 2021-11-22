FROM nvcr.io/nvidia/tensorflow:21.08-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y  &&\
    apt-get install -y build-essential libasound2-dev libjack-dev portaudio19-dev libsndfile1-dev &&\ 
    pip install magenta &&\
    mkdir -m 777 /working && \
    cd /working &&\
    git clone https://github.com/tensorflow/magenta.git &&\
    chmod -R 777 magenta &&\
    cd magenta &&\
    pip install -e . &&\
    mkdir -m 777 /tmp/numba_cache

ENV NUMBA_CACHE_DIR=/tmp/numba_cache

EXPOSE 8888
WORKDIR /working