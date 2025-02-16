# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all application files
COPY common.py metrics.py mitigation.py ./
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports FastAPI runs on
EXPOSE 8000 8001 8002

# Start multiple FastAPI apps using a process manager
CMD ["sh", "-c", "uvicorn common:app --host 0.0.0.0 --port 8000 & \
                   uvicorn metrics:app --host 0.0.0.0 --port 8001 & \
                   uvicorn mitigation:app --host 0.0.0.0 --port 8002 && wait"]
