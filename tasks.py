import os
import io
import json
import datetime
import multiprocessing
import logging
import base64

from celery import Celery, group, chord
from celery.utils.log import get_task_logger
import pytesseract
from PIL import Image
import fitz        # PyMuPDF
import openai
import requests
from concurrent.futures import ThreadPoolExecutor

# ─── Configuración Celery ──────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("infovital", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.task_default_queue = "default"
logger = get_task_logger(__name__)
logger.setLevel(logging.INFO)

# ─── Configuración OpenAI ──────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Falta la variable OPENAI_API_KEY")

# ─── Constantes ────────────────────────────────────────────────────────────────
MAX_POLICY_CHARS  = 15000
MAX_SUMMARY_CHARS = 2000
now = lambda: datetime.datetime.now().strftime("%H:%M:%S")

# ─── Funciones de extracción y llamadas a OpenAI ──────────────────────────────
def extract_page_text(page_idx: int, pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(page_idx)
        txt = page.get_text().strip()
        doc.close()
        if txt:
            return txt
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img = Image.open(io.BytesIO(pix.tobytes()))
        return pytesseract.image_to_string(img, lang="spa", config="--psm 6").strip()
    except Exception as e:
        return f"[ERROR página {page_idx+1}: {e}]"

def extract_text_batch(pdf_bytes: bytes, inicio: int, fin: int) -> str:
    primero = inicio - 1
    total   = fin - primero
    resultados = [None] * total

    def tarea(i):
        resultados[i] = extract_page_text(primero + i, pdf_bytes)

    workers = max(1, multiprocessing.cpu_count() - 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(total):
            executor.submit(tarea, i)

    return "\n".join(resultados)

def call_openai_parrafos(texto: str) -> list:
    prompt = f"Extrae párrafos relevantes:\n\"\"\"\n{texto[:25000]}\n\"\"\n"
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=1500
    )
    content = resp.choices[0].message.content.strip()
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1]).strip()
    try:
        return json.loads(content)
    except:
        return [{"error":"No se pudo parsear JSON","raw":content}]

def call_openai_resumen(res_parc: list) -> str:
    bloque = "\n".join(f"- {r}" for r in res_parc if r)
    prompt = f"A partir de estos resúmenes:\n{bloque}\n\nGenera un resumen general en 2–3 párrafos."
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=1000
    )
    return resp.choices[0].message.content.strip()

def call_openai_clasificacion(texto_politica: str, resumen: str) -> dict:
    policy_excerpt  = texto_politica[:MAX_POLICY_CHARS]
    summary_excerpt = resumen[:MAX_SUMMARY_CHARS]
    prompt = (
        f"Política (fragmento):\n\"\"\"\n{policy_excerpt}\n\"\"\"\n\n"
        f"Resumen (fragmento):\n\"\"\"\n{summary_excerpt}\n\"\"\"\n\n"
        "Devuelve un JSON con { \"alineacion\":\"alta|media|baja\","
        " \"objetivos_alineados\":[...] }"
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=500
    )
    content = resp.choices[0].message.content.strip()
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1]).strip()
    try:
        return json.loads(content)
    except:
        return {"error":"No se pudo parsear clasificación","raw":content}

# ─── Subtarea: procesa un lote de páginas ───────────────────────────────────────
@celery_app.task(name="infovital.extract_batch")
def extract_batch_task(pdf_bytes_b64: str, inicio: int, fin: int) -> list:
    pdf_bytes = base64.b64decode(pdf_bytes_b64)
    try:
        texto = extract_text_batch(pdf_bytes, inicio, fin)
        return call_openai_parrafos(texto)
    except Exception as e:
        logger.exception("Error en extract_batch_task")
        return [{"error": str(e)}]

# ─── Callback final: resume, clasifica y notifica ───────────────────────────────
@celery_app.task(name="infovital.finalize", bind=True)
def finalize(self, results: list, pdf_bytes_b64: str, municipio: str, webhook_url: str):
    task_id = self.request.id
    logger.info(f"[{now()}] finalize arrancó para task {task_id} con {len(results)} batches")

    # Aplana resultados
    all_pars = [p for batch in results for p in (batch if isinstance(batch, list) else [])]

    # Resumen general
    res_parc = [p.get("resumen","") for p in all_pars if p.get("resumen")]
    resumen_general = call_openai_resumen(res_parc) if res_parc else ""

    # Textos de políticas
    ruta_cu = os.getenv("RUTA_TXT_CUIDADO", "")
    ruta_ev = os.getenv("RUTA_TXT_ENVEJECIMIENTO", "")
    text_cu = open(ruta_cu, "r", encoding="utf-8").read() if ruta_cu else ""
    text_ev = open(ruta_ev, "r", encoding="utf-8").read() if ruta_ev else ""

    # Clasificaciones
    clas_cu = call_openai_clasificacion(text_cu, resumen_general)
    clas_ev = call_openai_clasificacion(text_ev, resumen_general)

    payload = {
        "task_id": task_id,
        "municipio": municipio,
        "total_parrafos": len(all_pars),
        "resultados_parrafos": all_pars,
        "resumen_general": resumen_general,
        "clasificacion_cuidado": clas_cu,
        "clasificacion_envejecimiento": clas_ev,
        "link_al_archivo": ""
    }

    try:
        r = requests.post(webhook_url, json=payload, timeout=30)
        r.raise_for_status()
        logger.info(f"[{now()}] Webhook OK para {task_id}")
    except Exception as e:
        logger.exception(f"Error al enviar webhook para {task_id}: {e}")

# ─── Tarea maestra: divide en batches y dispara chord ───────────────────────────
@celery_app.task(name="infovital.process_pdf", bind=True)
def process_pdf_task(self, pdf_bytes_b64: str, municipio: str, webhook_url: str):
    task_id = self.request.id
    logger.info(f"[{now()}] process_pdf_task arrancó, task_id={task_id}")

    pdf_bytes = base64.b64decode(pdf_bytes_b64)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    doc.close()

    batches = [
        extract_batch_task.s(pdf_bytes_b64, start, min(start+49, total))
        for start in range(1, total+1, 50)
    ]
    chord(group(batches), finalize.s(pdf_bytes_b64, municipio, webhook_url)).delay()

    logger.info(f"[{now()}] Chord lanzado con {len(batches)} jobs para task {task_id}")
