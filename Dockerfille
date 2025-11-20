# Use a lightweight Python image
FROM python:3.11-slim

# 1. Install system dependencies (FFmpeg is crucial here)
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential && \
    rm -rf /var/lib/apt/lists/*

# 2. Set up the app directory
WORKDIR /app

# 3. Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code
COPY . .

# 5. Expose the port (Render sets PORT env var, but good to document)
EXPOSE 8000

# 6. Command to run the app
# We use the shell form to ensure $PORT variable is expanded correctly
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
