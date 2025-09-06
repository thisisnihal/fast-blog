FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files (except files in .dockerignore)
COPY . .

# Expose port for Uvicorn
EXPOSE 8000

# Run the FastAPI app
CMD ["python3", "main.py"]
