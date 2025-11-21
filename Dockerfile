# Use a lightweight Python image
FROM python:3.11-slim

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential && \
    rm -rf /var/lib/apt/lists/*

# --- NEW LINE: Force logs to show up immediately ---
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

# 6. Run the app (Make sure 'app:app' matches your filename 'app.py')
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"

