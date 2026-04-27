# Dockerfile para desplegar el modelo en cualquier servidor con GPU
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Instalar dependencias
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Compilar llama.cpp con CUDA
WORKDIR /app
RUN git clone https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && make LLAMA_CUDA=1

# Copiar modelo y mmproj
COPY Checkpoint-375-Ollama-Clean-7.5B-Q4_K_M.gguf /app/models/
COPY mmproj-Checkpoint-375-Ollama-Clean-BF16.gguf /app/models/

# Exponer puerto
EXPOSE 8081

# Comando de inicio
CMD ["/app/llama.cpp/llama-server", \
     "-m", "/app/models/Checkpoint-375-Ollama-Clean-7.5B-Q4_K_M.gguf", \
     "--mmproj", "/app/models/mmproj-Checkpoint-375-Ollama-Clean-BF16.gguf", \
     "--host", "0.0.0.0", \
     "--port", "8081", \
     "-c", "16384", \
     "-ngl", "99", \
     "--jinja"]