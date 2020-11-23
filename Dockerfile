FROM wekaco/pytorch-cuda:latest

# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
   pip install --no-cache-dir -r requirements.txt

WORKDIR /app

RUN mkdir -p datasets && mkdir -p results

COPY LICENSE /app/LICENSE
COPY README.md /app/README.md

COPY *.py /app/
COPY trainer /app/trainer
COPY gen /app/gen
