FROM anibali/pytorch:cuda-9.1

RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
  ffmpeg \
  libav-tools \
  libavcodec-extra \
  && sudo rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && \
   pip install -r requirements.txt

ENTRYPOINT [ "python", "train.py" ]

WORKDIR /app

RUN mkdir -p datasets && mkdir -p results

COPY LICENSE /app/LICENSE
COPY README.md /app/README.md

COPY *.py /app/
COPY trainer /app/trainer
