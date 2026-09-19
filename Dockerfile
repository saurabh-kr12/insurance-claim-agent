# Dockerfile
# ----------
# Optional containerized deployment. Note: this container talks to Ollama
# running on the HOST machine (see OLLAMA_BASE_URL below) rather than
# bundling an LLM inside the image, keeping the image small.
#
# Build:  docker build -t claims-rag-agent .
# Run:    docker run -p 8501:8501 --env-file .env claims-rag-agent
#         (On Linux, set OLLAMA_BASE_URL=http://host.docker.internal:11434
#          or use --network host so the container can reach your local Ollama.)

FROM python:3.11-slim

WORKDIR /app

# System deps needed by some packages (e.g. building sentence-transformers deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Streamlit needs 0.0.0.0 to be reachable from outside the container.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
