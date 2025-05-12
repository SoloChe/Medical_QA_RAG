FROM --platform=linux/x86-64 python:3.10-slim


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY ../requirements.txt .

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
    pip install -r requirements.txt && \
        rm -rf /root/.cache

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV MODEL_PATH="/opt/ml/model"

COPY inference /opt/program
WORKDIR /opt/program

RUN chmod +x serve



