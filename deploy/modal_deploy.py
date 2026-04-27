"""
Deploy Clarity E4B Vision to Modal (serverless GPU).
Install: pip install modal
Deploy: modal deploy modal_deploy.py
"""

import modal

# Crear app
app = modal.App("clarity-e4b-vision")

# Imagen con llama.cpp
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("git", "build-essential", "curl")
    .run_commands(
        "cd /root && git clone https://github.com/ggml-org/llama.cpp.git",
        "cd /root/llama.cpp && make LLAMA_CUDA=1",
    )
)

# Volumen para el modelo
volume = modal.Volume.from_name("clarity-models", create_if_missing=True)
MODEL_PATH = "/models"
MMPROJ_NAME = "mmproj-Checkpoint-375-Ollama-Clean-BF16.gguf"
MODEL_NAME = "Checkpoint-375-Ollama-Clean-7.5B-Q4_K_M.gguf"


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
                "/root/llama.cpp/llama-server",
                "-m", f"{MODEL_PATH}/{MODEL_NAME}",
                "--mmproj", f"{MODEL_PATH}/{MMPROJ_NAME}",
                "--host", "0.0.0.0",
                "--port", "8080",
                "-c", "16384",
                "-ngl", "99",
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
        return {"status": "ok", "model": "clarity-e4b-vision"}


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