import base64
import logging
from fastapi import FastAPI, UploadFile, File, Form
from tasks import process_pdf_task

app = FastAPI()
logger = logging.getLogger("uvicorn")

@app.post("/process_pdf/")
async def process_pdf(
    file: UploadFile = File(...),
    municipio: str = Form(...),
    webhook_url: str = Form(...)
):
    """
    Recibe el PDF, lo codifica en base64 y lanza la tarea Celery.
    Devuelve inmediatamente el task_id para seguimiento.
    """
    pdf_bytes = await file.read()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf8")

    result = process_pdf_task.delay(pdf_b64, municipio, webhook_url)
    logger.info(f"Tarea Celery lanzada con ID {result.id}")
    return {"task_id": result.id}
