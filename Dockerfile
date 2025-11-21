# Use a lightweight Python image
FROM python:3.11-slim

# 1. Install system dependencies
# ADDED: ca-certificates (Required for Edge-TTS to work)
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Force logs to show (Debugging)
ENV PYTHONUNBUFFERED=1

# 2. Set up the app directory
WORKDIR /app

# 3. Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code
COPY . .

# 5. Expose the port
EXPOSE 8000

# 6. Run the app
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"

