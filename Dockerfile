FROM python:3.12-slim

WORKDIR /toxic_comment_classification

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    gzip \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY ml/ ml/
COPY web_app/ web_app/
COPY scripts/ scripts/
COPY models/ models/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD [ "bash", "scripts/up.sh" ]