FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /usr/src/app

RUN apt update && apt install -y python3 python3-pip ffmpeg libsm6 libxext6 wget unzip
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -P /root/.insightface/models \
    && unzip /root/.insightface/models/buffalo_l.zip -d /root/.insightface/models/buffalo_l \
    && rm /root/.insightface/models/buffalo_l.zip && apt-get remove -y wget unzip

COPY src/ ./src
COPY scripts/ ./scripts

ENTRYPOINT ["/usr/bin/python3", "scripts/detect_faces.py"]
CMD ["--help"]

