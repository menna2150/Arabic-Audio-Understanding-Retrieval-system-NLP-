FROM python:3.12-slim

WORKDIR /app

# Install system deps (librosa needs ffmpeg)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
