FROM python:3.12

RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /usr/src/app
COPY . .
COPY .env ./
RUN pip install --no-cache-dir -r requirements.txt
