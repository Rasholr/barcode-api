FROM python:3.11-slim

# Install system library required for pyzbar (ZBar)
RUN apt-get update && apt-get install -y --no-install-recommends libzbar0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
ENV PORT=8000
EXPOSE 8000
CMD ["python", "app.py"]