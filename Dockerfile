# Use official Python image as base
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies

RUN apt-get update && apt-get install -y libgl1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port (match docker-compose)
EXPOSE 8000

# Set default command (adjust as needed for your framework)
CMD ["python", "-m", "uvicorn", "python_files.backend_fast:app", "--host", "0.0.0.0", "--port", "8000"]
