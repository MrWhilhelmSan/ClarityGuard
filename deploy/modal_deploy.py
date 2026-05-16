"""
Deploy Clarity E4B Vision to Modal (serverless GPU).
Install: pip install modal
Deploy: modal deploy modal_deploy.py
"""

import modal

# Crear app
app = modal.App("clarityguard-v2")

# Imagen con llama.cpp
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04")
    .apt_install("git", "cmake", "build-essential", "curl")
    .run_commands(
        "cd /root && git clone https://github.com/ggml-org/llama.cpp.git",
        "cd /root/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release",
        "cd /root/llama.cpp && cmake --build build -j --target llama-server",
    )
)

# Volumen para el modelo
volume = modal.Volume.from_name("clarity-models", create_if_missing=True)
MODEL_PATH = "/models"
MMPROJ_NAME = "mmproj-ClarityGuard-v2.gguf"
MODEL_NAME = "ClarityGuard-v2.gguf"


@app.cls(
    image=image,
    gpu="A10G",  # o "L4", "A100"
    volumes={MODEL_PATH: volume},
    timeout=300,
)
class ClarityE4B:
    @modal.enter()
    def load_model(self):
        """Cargar modelo al iniciar"""
        import subprocess
        self.server_process = subprocess.Popen(
            [
                "/root/llama.cpp/build/bin/llama-server",
                "-m", f"{MODEL_PATH}/{MODEL_NAME}",
                "--mmproj", f"{MODEL_PATH}/{MMPROJ_NAME}",
                "--host", "0.0.0.0",
                "--port", "8080",
                "-c", "12288",
                "-ngl", "999",
                "--jinja",
            ]
        )

    @modal.fastapi_endpoint(method="POST")
    def generate(self, prompt: str, image_url: str = None):
        """Generar respuesta"""
        import requests
        import time

        time.sleep(2)  # Esperar a que cargue el modelo

        payload = {
            "prompt": prompt,
            "n_predict": 256,
            "temperature": 1.0,
        }

        if image_url:
            payload["image"] = image_url

        response = requests.post(
            "http://localhost:8080/completion",
            json=payload,
            timeout=60,
        )
        return response.json()

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "ok", "model": "clarityguard-v2"}


# Comando para subir modelos (ejecutar una vez)
@app.function(volumes={MODEL_PATH: volume})
def upload_models():
    """Subir modelos al volumen de Modal"""
    import os
    print(f"Sube tus archivos GGUF a: {MODEL_PATH}")
    print(f"Necesitas: {MODEL_NAME} y {MMPROJ_NAME}")


# Local entrypoint para pruebas
@app.local_entrypoint()
def main():
    with ClarityE4B() as model:
        result = model.generate.remote("Hola, responde brevemente")
        print(result)
