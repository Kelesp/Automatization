# function_app.py

import os
import json
import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

import pytesseract
from PIL import Image
import fitz  # PyMuPDF

import openai
from concurrent.futures import ThreadPoolExecutor
import io
import multiprocessing

# ────────────────────────────────────────────────────────────────────────────────
# 0) Rutas a archivos de política (cuidado y envejecimiento)
# ────────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_TXT_CUIDADO = os.path.join(BASE_DIR, "politicas_txt", "politica_cuidado.txt")
RUTA_TXT_ENVEJECIMIENTO = os.path.join(BASE_DIR, "politicas_txt", "politica_envejecimiento.txt")

try:
    with open(RUTA_TXT_CUIDADO, "r", encoding="utf-8") as f:
        TEXT_CUIDADO = f.read()
    with open(RUTA_TXT_ENVEJECIMIENTO, "r", encoding="utf-8") as f:
        TEXT_ENVEJECIMIENTO = f.read()
except Exception as e:
    raise RuntimeError(f"No se pudo abrir archivo de política: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# 1) Configuración de OpenAI y FastAPI
# ────────────────────────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("No se encontró la variable de entorno OPENAI_API_KEY")

# Ajusta la ruta a tesseract si no está en el PATH (por ejemplo: /usr/bin/tesseract)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI(
    title="API de Extracción y Clasificación de Párrafos Relevantes",
    description="Recibe un PDF y devuelve JSON con párrafos relevantes, resúmenes y clasificaciones.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────────
# 2) Funciones auxiliares: OCR y extracción optimizada en memoria
# ────────────────────────────────────────────────────────────────────────────────

def extract_page_text(page_idx: int, pdf_bytes: bytes) -> str:
    """
    Procesa una sola página: primero intenta obtener texto nativo;
    si está vacío, lo renderiza a imagen y ejecuta OCR con Tesseract.
    Retorna un string con la etiqueta de página y el texto.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(page_idx)
        txt = page.get_text().strip()
        if txt:
            result = f"--- Página {page_idx+1} (texto nativo) ---\n{txt}\n"
        else:
            # Renderizar la página a imagen a resolución moderada (matrix=2,2 equivale a ~144 DPI)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes(output="png")
            img = Image.open(io.BytesIO(img_bytes))
            ocr_txt = pytesseract.image_to_string(
                img,
                lang="spa",
                config="--psm 6"
            ).strip()
            result = f"--- Página {page_idx+1} (OCR) ---\n{ocr_txt}\n"
        doc.close()
    except Exception as e:
        result = f"--- Página {page_idx+1}: ERROR al procesar: {e} ---\n\n"
    return result

def extract_text_batch(pdf_bytes: bytes, pagina_inicio: int, pagina_fin: int, num_workers: int = 4) -> str:
    """
    Extrae texto (nativo u OCR) de un rango de páginas en paralelo.
    - pagina_inicio y pagina_fin son 1-based inclusive.
    - num_workers: nº de threads en ThreadPoolExecutor.
    Devuelve un único string con el texto concatenado de todas esas páginas.
    """
    # Convertir a índice 0-based
    primero = pagina_inicio - 1
    ultimo = pagina_fin - 1
    total = ultimo - primero + 1

    # Array para guardar resultados de cada página
    resultados = [None] * total

    def tarea(idx_relative: int):
        page_idx = primero + idx_relative
        resultados[idx_relative] = extract_page_text(page_idx, pdf_bytes)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(total):
            executor.submit(tarea, i)

    # Concatenar con saltos de línea entre páginas
    return "\n".join(resultados)

# ────────────────────────────────────────────────────────────────────────────────
# Helper para timestamp en logs
# ────────────────────────────────────────────────────────────────────────────────
now = lambda: datetime.datetime.now().strftime("%H:%M:%S")

# ────────────────────────────────────────────────────────────────────────────────
# 3) Endpoint principal: /process_pdf/
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/process_pdf/")
async def process_pdf(
    file: UploadFile = File(..., description="Archivo PDF"),
    municipio: str = Form(..., description="Nombre del municipio"),
):
    # 3.1) Validar extensión y leer todo el PDF en memoria
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer PDF: {e}")

    # 3.2) Obtener total de páginas con PyMuPDF en memoria
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            total_paginas = len(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al abrir PDF en memoria: {e}")

    print(f"[{now()}] PDF recibido en memoria; total de páginas: {total_paginas}")

    # 3.3) Procesar en lotes (por ejemplo, 50 páginas por lote para no pasarse de tokens)
    lista_parrafos = []
    lote_size = 50  # Ajusta según la longitud de tus documentos y límites de tokens

    # Determinar núm. de hilos basándonos en CPUs disponibles
    cpus = multiprocessing.cpu_count()
    num_workers = max(1, cpus - 1)

    for inicio in range(1, total_paginas + 1, lote_size):
        fin = min(inicio + lote_size - 1, total_paginas)
        print(f"[{now()}] → Procesando lote páginas {inicio} a {fin}…")

        # 3.3.1) Extraer texto nativo/OCR en paralelo para este rango
        start_lote = datetime.datetime.now()
        try:
            texto_lote = extract_text_batch(pdf_bytes, inicio, fin, num_workers=num_workers)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error extrayendo texto (páginas {inicio}-{fin}): {e}"
            )
        elapsed_lote = (datetime.datetime.now() - start_lote).total_seconds()
        print(f"[{now()}] ---- Texto extraído (Lote {inicio}-{fin}) en {round(elapsed_lote, 1)}s.")

        fragmento_debug = texto_lote[:200] if texto_lote else ""
        print(f"[DEBUG {now()}] Fragmento (páginas {inicio}-{fin}): {fragmento_debug!r}\n")

        # 3.3.2) Limitar el texto a 25 000 caracteres para el prompt
        texto_para_prompt = texto_lote[:25000]

        # 3.3.3) Construir prompt para extraer párrafos relevantes
        prompt_parrafos = f"""
Extrae únicamente los párrafos sobre alguno de estos temas (en español):
• Personas mayores, Adulto mayor, Personas cuidadoras, Personas que requieren cuidado, Cuidadores, Sistema de cuidado, Caracterización de personas mayores, Caracterización de cuidadores.

Para cada párrafo devuélvelo como objeto JSON con campos exactos:
  - "pagina": número de página (1-based, en el documento completo)
  - "parrafo": texto completo del párrafo
  - "tema": cuál de los temas detectaste
  - "resumen": breve resumen del párrafo
  - "relevancia": "baja"|"media"|"alta"

Sin explicaciones adicionales. Devuelve solo un ARRAY JSON válido.

Texto (páginas {inicio}-{fin}, recortado a 25000 caracteres):
"""
        prompt_parrafos += texto_para_prompt

        print(f"[{now()}] Enviando prompt a OpenAI (gpt-3.5-turbo) para extraer párrafos (Páginas {inicio}-{fin})…")
        start_call = datetime.datetime.now()
        try:
            response1 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_parrafos}],
                temperature=0.0,
                max_tokens=1500,
            )
            response_text1 = response1.choices[0].message.content.strip()
        except openai.error.OpenAIError as e_openai:
            raise HTTPException(
                status_code=500,
                detail=f"Error OpenAI (párrafos, páginas {inicio}-{fin}): {e_openai}"
            )
        elapsed_call = (datetime.datetime.now() - start_call).total_seconds()
        print(f"[{now()}] ---- Respuesta OpenAI (párrafos) recibida en {round(elapsed_call, 1)}s.")

        # 3.3.4) Quitar delimitadores ``` en caso de que OpenAI los devuelva
        if response_text1.startswith("```") and response_text1.endswith("```"):
            lines = response_text1.splitlines()
            if len(lines) >= 3:
                response_text1 = "\n".join(lines[1:-1]).strip()

        # 3.3.5) Intentar parsear JSON; si falla, generar un elemento de error
        try:
            resultados_parciales = json.loads(response_text1)
        except json.JSONDecodeError:
            resultados_parciales = [{
                "pagina": None,
                "parrafo": None,
                "tema": None,
                "resumen": None,
                "relevancia": None,
                "error": "No se pudo parsear JSON",
                "raw_response": response_text1
            }]

        # 3.3.6) Acumular resultados
        lista_parrafos.extend(resultados_parciales)

    # ────────────────────────────────────────────────────────────────────────────────
    # 3.4) Generar Resumen General a partir de los resúmenes parciales
    # ────────────────────────────────────────────────────────────────────────────────
    print(f"[{now()}] Generando Resumen General a partir de {len(lista_parrafos)} ítems…")
    resumen_general = ""
    try:
        res_parciales = [
            item.get("resumen", "").strip()
            for item in lista_parrafos
            if isinstance(item, dict) and item.get("resumen")
        ]
        if res_parciales:
            bloque_para_resumen = "\n".join(f"- {r}" for r in res_parciales if r)
            prompt_resumen = f"""
A partir de estos resúmenes parciales:
{bloque_para_resumen}

Genera un Resumen General coherente en 2–3 párrafos.
Solo devuelve texto, sin formato adicional.
"""
            start_resumen = datetime.datetime.now()
            resp2 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_resumen}],
                temperature=0.0,
                max_tokens=1000,
            )
            resumen_general = resp2.choices[0].message.content.strip()
            elapsed_resumen = (datetime.datetime.now() - start_resumen).total_seconds()
            print(f"[{now()}] ---- Resumen General recibido en {round(elapsed_resumen, 1)}s.")
        else:
            resumen_general = "No hay resúmenes parciales para generar un Resumen General."
    except openai.error.OpenAIError as e_resumen:
        resumen_general = f"Error al generar Resumen General: {e_resumen}"

    # ────────────────────────────────────────────────────────────────────────────────
    # 3.5) Clasificar alineación contra Política de Cuidado
    # ────────────────────────────────────────────────────────────────────────────────
    print(f"[{now()}] Clasificando alineación contra Política de Cuidado…")
    clasificacion_cuidado = {}
    try:
        prompt_cuidado = f"""
Eres un experto en políticas públicas de cuidado. A continuación se te proporciona:

1) Texto completo de la Política Nacional de Cuidado:
\"\"\"\n{TEXT_CUIDADO}\n\"\"\"

2) Resumen General del municipio «{municipio}»:
\"\"\"\n{resumen_general}\n\"\"\"

A. Evalúa nivel de ALINEACIÓN (alta|media|baja).  
B. Especifica con qué objetivos de la política de cuidado se alinea (por ejemplo: ["Objetivo 1", "Objetivo 2", …]).

Solo devuelve un JSON con formato:
{{ 
  "alineacion": "alta|media|baja",
  "objetivos_alineados": ["Objetivo X", "Objetivo Y", …]
}}
Sin texto extra, sin explicaciones.
"""
        start_cuidado = datetime.datetime.now()
        resp3 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_cuidado}],
            temperature=0.0,
            max_tokens=500,
        )
        text3 = resp3.choices[0].message.content.strip()
        if text3.startswith("```") and text3.endswith("```"):
            lines = text3.splitlines()
            if len(lines) >= 3:
                text3 = "\n".join(lines[1:-1]).strip()
        clasificacion_cuidado = json.loads(text3)
        elapsed_cuidado = (datetime.datetime.now() - start_cuidado).total_seconds()
        print(f"[{now()}] ---- Clasificación Cuidado recibida en {round(elapsed_cuidado, 1)}s.")
    except Exception as e_cu:
        clasificacion_cuidado = {"error": f"No se pudo clasificar Cuidado: {e_cu}"}

    # ────────────────────────────────────────────────────────────────────────────────
    # 3.6) Clasificar alineación contra Política de Envejecimiento y Vejez
    # ────────────────────────────────────────────────────────────────────────────────
    print(f"[{now()}] Clasificando alineación contra Política de Envejecimiento…")
    clasificacion_envejecimiento = {}
    try:
        prompt_envejecimiento = f"""
Eres un experto en políticas de envejecimiento y vejez. A continuación se te proporciona:

1) Texto completo de la Política Nacional de Envejecimiento y Vejez:
\"\"\"\n{TEXT_ENVEJECIMIENTO}\n\"\"\"

2) Resumen General del municipio «{municipio}»:
\"\"\"\n{resumen_general}\n\"\"\"

A. Evalúa nivel de ALINEACIÓN (alta|media|baja).  
B. Especifica con qué objetivos de la política de envejecimiento se alinea (por ejemplo: ["Objetivo X", "Objetivo Y", …]).

Solo devuelve un JSON con formato:
{{ 
  "alineacion": "alta|media|baja",
  "objetivos_alineados": ["Objetivo A", "Objetivo B", …]
}}
Sin texto extra, sin explicaciones.
"""
        start_enve = datetime.datetime.now()
        resp4 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_envejecimiento}],
            temperature=0.0,
            max_tokens=500,
        )
        text4 = resp4.choices[0].message.content.strip()
        if text4.startswith("```") and text4.endswith("```"):
            lines = text4.splitlines()
            if len(lines) >= 3:
                text4 = "\n".join(lines[1:-1]).strip()
        clasificacion_envejecimiento = json.loads(text4)
        elapsed_enve = (datetime.datetime.now() - start_enve).total_seconds()
        print(f"[{now()}] ---- Clasificación Envejecimiento recibida en {round(elapsed_enve, 1)}s.")
    except Exception as e_en:
        clasificacion_envejecimiento = {"error": f"No se pudo clasificar Envejecimiento: {e_en}"}

    # 3.7) Respuesta final
    return {
        "municipio": municipio,
        "total_paginas": total_paginas,
        "resultados_parrafos": lista_parrafos,
        "resumen_general": resumen_general,
        "clasificacion_cuidado": clasificacion_cuidado,
        "clasificacion_envejecimiento": clasificacion_envejecimiento,
    }

# ────────────────────────────────────────────────────────────────────────────────
# 4) Arrancar con Uvicorn si lo ejecutamos directamente (solo para desarrollo)
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("function_app:app", host="0.0.0.0", port=8002, reload=True)
