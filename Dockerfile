FROM ubuntu:focal

RUN apt-get update -qq && \
    apt-get install -y unzip locales && \
    apt-get clean && \ 
    rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 && update-locale

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libx11-6 \
    gdebi-core \
    libapparmor1  \
    libcurl4-openssl-dev \
    build-essential \
    gnupg2 \
    cmake \
    curl \ 
    && rm -rf /var/lib/apt/lists/*


ENV CONDA_AUTO_UPDATE_CONDA=false
RUN curl -sLo ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /usr/conda && \
    rm ~/miniconda.sh

ENV PATH=/usr/conda/bin:$PATH

COPY ./requirements.txt requirements.txt
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt  \
    && rm -rf /root/.cache

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

COPY ./weights.hdf5 weights.hdf5
COPY ./test.py test.py

COPY ./preprocessor.py preprocessor.py
COPY ./model.py model.py
COPY ./run.sh run.sh
COPY ./sample_test sample_test

CMD ["bash" , "./run.sh"]