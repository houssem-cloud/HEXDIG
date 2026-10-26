# Optional: Docker alternative to the systemd/setup.sh route.
# docker compose up -d --build
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "-u", "main.py"]
