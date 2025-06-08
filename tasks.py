# tasks.py

import os
import io
import json
import datetime
import multiprocessing
from celery import Celery
import pytesseract
from PIL import Image
import fitz      # PyMuPDF
import openai
import requests

# ────────────────────────────────────────────────────────────────────────────────
# 1) Configuración de Celery con Redis
# ────────────────────────────────────────────────────────────────────────────────
REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app   = Celery("infovital_tasks", broker=REDIS_URL)

# ────────────────────────────────────────────────────────────────────────────────
# 2) Configuración de OpenAI
# ────────────────────────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Falta la variable OPENAI_API_KEY")

# Helper de timestamp para logs
now = lambda: datetime.datetime.now().strftime("%H:%M:%S")

# ────────────────────────────────────────────────────────────────────────────────
# 3) Funciones de extracción y llamadas a OpenAI
# ────────────────────────────────────────────────────────────────────────────────

def extract_page_text(page_idx: int, pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(page_idx)
        txt = page.get_text().strip()
        if txt:
            res = f"--- Página {page_idx+1} (texto nativo) ---\n{txt}\n"
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_txt = pytesseract.image_to_string(img, lang="spa").strip()
            res = f"--- Página {page_idx+1} (OCR) ---\n{ocr_txt}\n"
        doc.close()
    except Exception as e:
        res = f"--- Página {page_idx+1}: ERROR {e} ---\n"
    return res

def extract_text_batch(pdf_bytes: bytes, inicio: int, fin: int) -> str:
    primero = inicio - 1
    indices = list(range(primero, fin))
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        textos = pool.map(lambda i: extract_page_text(i, pdf_bytes), indices)
    return "\n".join(textos)

def call_openai_parrafos(texto: str, inicio: int, fin: int) -> list:
    prompt = f"""
Extrae únicamente los párrafos sobre estos temas (en español):
Texto (páginas {inicio}-{fin}):
\"\"\"\n{texto[:25000]}\n\"\"\"
"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=1500,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        return json.loads(text)
    except:
        return [{"error":"No JSON","raw":text}]

def call_openai_resumen(res_parc: list) -> str:
    bloque = "\n".join(f"- {r}" for r in res_parc if r)
    prompt = f"""
A partir de estos resúmenes parciales:
{bloque}

Genera un Resumen General coherente en 2–3 párrafos.
"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=1000,
    )
    return resp.choices[0].message.content.strip()

def call_openai_clasificacion(texto_política: str, resumen: str) -> dict:
    prompt = f"""
Eres un experto en políticas. Política:
\"\"\"\n{texto_política}\n\"\"\"
Resumen General:
\"\"\"\n{resumen}\n\"\"\"

Devuelve JSON: {{ "alineacion":"alta|media|baja", "objetivos_alineados":[…] }}
"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        return json.loads(text)
    except:
        return {"error":"No pudo parsear clasificación","raw":text}

# ────────────────────────────────────────────────────────────────────────────────
# 4) Tarea principal de Celery
# ────────────────────────────────────────────────────────────────────────────────

@celery_app.task(name="infovital.process_pdf", bind=True, acks_late=True)
def process_pdf_task(self, pdf_bytes: bytes, municipio: str, webhook_url: str):
    task_id = self.request.id
    try:
        # Contar páginas
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_paginas = len(doc)
        doc.close()
        print(f"[{now()}] Tarea {task_id}: {municipio} → {total_paginas} págs")

        # Extraer y GPT por lotes
        lista = []
        for inicio in range(1, total_paginas+1, 50):
            fin = min(inicio+49, total_paginas)
            texto = extract_text_batch(pdf_bytes, inicio, fin)
            parc = call_openai_parrafos(texto, inicio, fin)
            lista.extend(parc)

        # Resumen general
        parciales = [p.get("resumen","") for p in lista if p.get("resumen")]
        resumen = call_openai_resumen(parciales) if parciales else ""

        # Clasificaciones
        text_cu = open(os.getenv("RUTA_TXT_CUIDADO"),"r",encoding="utf-8").read()
        text_ev = open(os.getenv("RUTA_TXT_ENVEJECIMIENTO"),"r",encoding="utf-8").read()
        cls_cu = call_openai_clasificacion(text_cu, resumen)
        cls_ev = call_openai_clasificacion(text_ev, resumen)

        # JSON final
        result = {
            "task_id": task_id,
            "municipio": municipio,
            "total_paginas": total_paginas,
            "resultados_parrafos": lista,
            "resumen_general": resumen,
            "clasificacion_cuidado": cls_cu,
            "clasificacion_envejecimiento": cls_ev,
            "link_al_archivo": ""
        }

        # Callback a n8n
        r = requests.post(webhook_url, json=result, timeout=30)
        r.raise_for_status()
        print(f"[{now()}] Tarea {task_id}: callback OK {r.status_code}")

    except Exception as e:
        err = {"task_id": task_id, "error": str(e), "municipio": municipio}
        try:
            requests.post(webhook_url, json=err, timeout=10)
        except:
            pass
        raise
