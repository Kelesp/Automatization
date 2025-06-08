# function_app.py

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Importamos la tarea de Celery
from tasks import process_pdf_task

# ────────────────────────────────────────────────────────────────────────────────
# 0) Cargar variables de entorno
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
if not N8N_WEBHOOK_URL:
    raise RuntimeError("Falta la variable de entorno N8N_WEBHOOK_URL")

# ────────────────────────────────────────────────────────────────────────────────
# 1) Crear la app FastAPI
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="API de Extracción y Encolado",
    description="Recibe un PDF y un municipio, encola la tarea en Celery y responde rápido.",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# 2) Endpoint para encolar el PDF
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/process_pdf/")
async def process_pdf(
    file: UploadFile = File(..., description="Archivo PDF"),
    municipio: str    = Form(...,       description="Nombre del municipio"),
):
    # Validar extensión
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

    # Leer el PDF en memoria
    contenido = await file.read()

    # Encolar la tarea en Celery
    task = process_pdf_task.delay(contenido, municipio, N8N_WEBHOOK_URL)

    # Responder inmediatamente
    return {
        "status":    "queued",
        "task_id":   task.id
    }

# ────────────────────────────────────────────────────────────────────────────────
# 3) Para correr en desarrollo con autoreload
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "function_app:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        workers=1,
    )
