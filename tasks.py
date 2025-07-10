import os
import io
import json
import datetime
import logging
import fitz
from PIL import Image
import pytesseract
import openai
import requests
from celery import Celery
from celery.utils.log import get_task_logger
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Configuración de Celery ────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("infovital", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.worker_prefetch_multiplier = 1

logger = get_task_logger(__name__)
logger.setLevel(logging.INFO)
now = lambda: datetime.datetime.now().strftime("%H:%M:%S")

# ─── Configuración de OpenAI ────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Falta OPENAI_API_KEY")

# ─── Palabras clave ─────────────────────────────────────────────────────────────
KEYWORDS = [
    "Personas mayores", "Adultos mayores", "Cuidadores",
    "Sistemas de cuidado", "Caracterización de cuidadores",
    "Caracterización de personas mayores"      
]

# ─── Objetivos oficiales ────────────────────────────────────────────────────────
OBJ_PATH = os.getenv("RUTA_OBJETIVOS_JSON", "objetivos.json")
if os.path.exists(OBJ_PATH):
    with open(OBJ_PATH, "r", encoding="utf-8") as f:
        POSSIBLE_OBJECTIVES = json.load(f)
else:
    POSSIBLE_OBJECTIVES = []

# ─── Funciones de extracción ─────────────────────────────────────────────────────
def extract_text_native(pdf_bytes: bytes, page_index: int) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc.load_page(page_index).get_text("text") or ""
    doc.close()
    return text.strip()

def ocr_page(pdf_bytes: bytes, page_index: int) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc.load_page(page_index).get_pixmap(matrix=fitz.Matrix(2, 2))
    doc.close()
    img = Image.open(io.BytesIO(pix.tobytes()))
    return pytesseract.image_to_string(img, lang="spa", config="--psm 4").strip()

def extract_and_filter(args):
    page_index, pdf_bytes = args
    txt = extract_text_native(pdf_bytes, page_index)
    if not txt:
        txt = ocr_page(pdf_bytes, page_index)
    lower = txt.lower()
    for kw in KEYWORDS:
        if kw.lower() in lower:
            return {"pagina": page_index + 1, "parrafo": txt, "tema_detectado": kw}
    return None

# ─── OpenAI: análisis por párrafo en paralelo ───────────────────────────────────
def process_parrafo(p):
    bloque = f"{p['parrafo'][:3000]}"
    tema = p['tema_detectado']
    pagina = p['pagina']
    prompt = f"""
Del siguiente texto detectado en la página {pagina} con tema “{tema}”, genera:

- "resumen": resumen del párrafo
- "relevancia": alta, media o baja, basándote en la importancia del tema con respecto a los objetivos de desarrollo social

Devuélvelo en JSON como:
{{
  "pagina": {pagina},
  "tema_detectado": "{tema}",
  "parrafo": "...",
  "resumen": "...",
  "relevancia": "..."
}}

Texto:
\"\"\"\n{bloque}\n\"\"\""""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )
        content = resp.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"❌ Error OpenAI página {pagina}: {e}")
        return None

def parallel_openai_parrafos(relevantes):
    with ThreadPoolExecutor(max_workers=4) as exe:
        results = list(exe.map(process_parrafo, relevantes))
    return [r for r in results if r]

# ─── OpenAI: resumen general, alineación, objetivos ─────────────────────────────
def call_openai_resumen(res_parc: list) -> str:
    bloque = "\n".join(f"- {r}" for r in res_parc if r)
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content":
            f"A partir de estos resúmenes:\n{bloque}\n\nGenera un resumen general en 2–3 párrafos."}],
        temperature=0.0,
        max_tokens=1000,
    )
    return resp.choices[0].message.content.strip()

def call_openai_clasificacion(resumen_general: str) -> str:
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content":
            f"Resumen general:\n\"\"\"\n{resumen_general[:2000]}\n\"\"\"\n\n"
            "Devuélveme solo uno de: alta, media o baja, basado en la alineación con los objetivos sociales de desarrollo."}],
        temperature=0.0,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip().lower()

def call_openai_objetivos(resumen_general: str) -> list[str]:
    posibles = "\n".join(f"- {o}" for o in POSSIBLE_OBJECTIVES)
    prompt = f"""
Dado este resumen general:
\"\"\"\n{resumen_general[:2000]}\n\"\"\"\

Y esta lista de objetivos de política:
{posibles}

Devuélveme JSON con los objetivos que aparecen en el resumen (array de strings). Si ninguno coincide, [].
"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    try:
        return json.loads(text)
    except:
        return []

# ─── Task principal ─────────────────────────────────────────────────────────────
@celery_app.task(name="infovital.process_pdf", bind=True)
def process_pdf_task(self, pdf_bytes, municipio: str, webhook_url: str):
    import base64
    if isinstance(pdf_bytes, str):
        pdf_bytes = base64.b64decode(pdf_bytes)

    task_id = self.request.id
    logger.info(f"[{now()}] process_pdf_task arrancó, task_id={task_id}")

    # total de páginas
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = doc.page_count
    doc.close()

    # extracción y filtrado en paralelo
    args = [(i, pdf_bytes) for i in range(total)]
    relevantes = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = {exe.submit(extract_and_filter, arg): arg[0] for arg in args}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res: relevantes.append(res)
            except Exception as e:
                logger.warning(f"[hilo] página falló: {e}")

    # si no hay nada relevante
    if not relevantes:
        payload = {
            "municipio": municipio,
            "resumen_general": "",
            "clasificacion_cuidado": {"alineacion": "", "objetivos_alineados": []},
            "resultados_parrafos": [],
            "link_al_archivo": ""
        }
        try:
            r = requests.post(webhook_url, json=payload, timeout=30)
            r.raise_for_status()
            logger.info(f"[{now()}] Webhook OK (vacío) para task {task_id}")
        except Exception as e:
            logger.exception(f"Error webhook vacío para task {task_id}: {e}")
        return

    # OpenAI: cada párrafo
    parrafos = parallel_openai_parrafos(relevantes)
    resumen_general = call_openai_resumen([p.get("resumen", "") for p in parrafos])
    alineacion = call_openai_clasificacion(resumen_general)
    objetivos = call_openai_objetivos(resumen_general)

    # Payload final
    payload = {
        "municipio": municipio,
        "resumen_general": resumen_general,
        "clasificacion_cuidado": {
            "alineacion": alineacion,
            "objetivos_alineados": objetivos
        },
        "resultados_parrafos": parrafos,
        "link_al_archivo": ""
    }

    try:
        r = requests.post(webhook_url, json=payload, timeout=30)
        r.raise_for_status()
        logger.info(f"[{now()}] Webhook OK para task {task_id}")
    except Exception as e:
        logger.exception(f"Error webhook final para task {task_id}: {e}")
