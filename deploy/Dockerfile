# Dockerfile para desplegar ClarityGuard v2 en un servidor con GPU
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Instalar dependencias
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Compilar llama.cpp con CUDA
WORKDIR /app
RUN git clone https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j --target llama-server

# Copiar modelo y projector multimodal.
# Descarga ambos desde:
# https://huggingface.co/CharlieBonito/clarity-guard-gemma4-7b
COPY ClarityGuard-v2.gguf /app/models/
COPY mmproj-ClarityGuard-v2.gguf /app/models/

# Exponer puerto
EXPOSE 8081

# Comando de inicio
CMD ["/app/llama.cpp/build/bin/llama-server", \
     "-m", "/app/models/ClarityGuard-v2.gguf", \
     "--mmproj", "/app/models/mmproj-ClarityGuard-v2.gguf", \
     "--host", "0.0.0.0", \
     "--port", "8081", \
     "-c", "12288", \
     "-ngl", "999", \
     "--jinja"]
