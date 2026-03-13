FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Do not copy the code here since we use a volume mount in docker-compose
# This way, changes on your local machine reflect instantly.

CMD ["tail", "-f", "/dev/null"]
