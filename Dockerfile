# Use slim Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000
EXPOSE 8501

# Run both servers in parallel
CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run front.py --server.port 8501 --server.enableCORS false"]
