FROM tensorflow/tensorflow:1.11.0-gpu-py3

LABEL maintainer=luozhouyang<stupidme.me.lzy@gmail.com>

COPY deepseg /root/deepseg
ENV PYTHONPATH=$PYTHONPATH:/root/deepseg

WORKDIR /root/deepseg