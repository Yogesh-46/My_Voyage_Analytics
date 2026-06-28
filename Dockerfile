# Use official Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask API port
EXPOSE 5001

# Start the Flask API
CMD ["python", "Src/Flight_Prediction/api.py"]