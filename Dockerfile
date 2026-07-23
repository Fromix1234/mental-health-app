FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir llama-cpp-python

COPY deploy_gguf.py .
COPY models/ ./models/

ENV PORT=8765
EXPOSE 8765

CMD ["python", "deploy_gguf.py"]
