# Copyright: (c) 2024, Amsterdam University Medical Centers
# Apache License, Version 2.0, (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0)

FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

# Set non-interactive installation to avoid getting stuck on timezone configuration
ENV DEBIAN_FRONTEND=noninteractive

# Update system and install dependencies
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Python package management and basic dependencies
RUN apt-get install -y curl python3.8 python3.8-dev python3.8-distutils

# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.8

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py --force-reinstall && rm get-pip.py

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output && chown algorithm:algorithm /opt/algorithm /input /output
RUN apt-get -y install libgtk-3-0

# Installing additional dependencies for OpenCV
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

RUN apt-get autoremove -y && apt-get autoclean

USER algorithm

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/

RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm scripts/predictor.py /opt/algorithm/

RUN mkdir -p /opt/algorithm/checkpoints
COPY --chown=algorithm:algorithm checkpoints/best.pth /opt/algorithm/checkpoints/

COPY --chown=algorithm:algorithm README.md /opt/algorithm/

ENTRYPOINT ["python", "predictor.py"]

## ALGORITHM LABELS ##

# These labels are required
LABEL qurai.amsterdam.algorithm.name=heartboundingboxestimation

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL qurai.amsterdam.algorithm.hardware.cpu.count=1
LABEL qurai.amsterdam.algorithm.hardware.cpu.capabilities=()
LABEL qurai.amsterdam.algorithm.hardware.memory=1G
LABEL qurai.amsterdam.algorithm.hardware.gpu.count=1
LABEL qurai.amsterdam.algorithm.hardware.gpu.cuda_compute_capability=8.6
LABEL qurai.amsterdam.algorithm.hardware.gpu.memory=3G
