# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn faiss-cpu tensorflow opencv-python numpy pillow

# Expose port 8000
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "classifier:app", "--host", "0.0.0.0", "--port", "8000"]
