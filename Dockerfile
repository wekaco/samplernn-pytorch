FROM anibali/pytorch:cuda-9.1

RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
  ffmpeg \
  libav-tools \
  youtube-dl \
  && sudo rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt && rm requirements.txt

ENTRYPOINT [ "python", "train.py" ]
