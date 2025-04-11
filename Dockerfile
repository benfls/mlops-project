FROM python:3.9-slim

WORKDIR /app

# Copy all necessary files into the container
COPY requirements.txt .
COPY src/ ./src/

RUN pip install -r requirements.txt

CMD ["python", "src/training.py"]
