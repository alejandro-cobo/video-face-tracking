FROM nvcr.io/nvidia/driver:525.60.13-ubuntu22.04
WORKDIR /usr/src/app
ENV HOME=/usr/src/app
COPY setup_cuda.sh ./
COPY src/ src/
COPY scripts/ scripts/
RUN apt update && apt install -y python3 python3-pip ffmpeg libsm6 libxext6 wget unzip && \
    pip install --no-cache -e . && \
    wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -P /usr/src/app/.insightface/models && \
    unzip /usr/src/app/.insightface/models/buffalo_l.zip -d /usr/src/app/.insightface/models/buffalo_l && \
    rm /usr/src/app/.insightface/models/buffalo_l.zip && \
    apt-get remove -y wget unzip && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    printf '#!/bin/bash\nsource setup_cuda.sh\n/usr/bin/python3 scripts/detect_faces.py $@' > run.sh && \
    chmod +x run.sh && \
    useradd -m appuser && \
    chown -R appuser /usr/src/app
USER appuser
ENTRYPOINT [ "/usr/src/app/run.sh" ]
CMD [ "--help" ]

