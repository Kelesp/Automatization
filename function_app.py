# function_app.py

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import fitz   # PyMuPDF
import tempfile
import openai
import json
import time

# ------------------------------------------------------------
# 0) Definir rutas a los TXT precargados de las políticas
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_TXT_CUIDADO = os.path.join(BASE_DIR, "politicas_txt", "politica_cuidado.txt")
RUTA_TXT_ENVEJECIMIENTO = os.path.join(BASE_DIR, "politicas_txt", "politica_envejecimiento.txt")

try:
    with open(RUTA_TXT_CUIDADO, "r", encoding="utf-8") as f:
        TEXT_CUIDADO = f.read()
except Exception as e:
    raise RuntimeError(f"No se pudo abrir {RUTA_TXT_CUIDADO}: {e}")

try:
    with open(RUTA_TXT_ENVEJECIMIENTO, "r", encoding="utf-8") as f:
        TEXT_ENVEJECIMIENTO = f.read()
except Exception as e:
    raise RuntimeError(f"No se pudo abrir {RUTA_TXT_ENVEJECIMIENTO}: {e}")

# ------------------------------------------------------------
# 1) Configuraciones iniciales de FastAPI y OpenAI
# ------------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("No se encontró la variable de entorno OPENAI_API_KEY")

# Ajusta la ruta a tesseract si es necesario (en Linux normalmente ya está en /usr/bin/tesseract;
# en Windows pon la ruta a tesseract.exe)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(
    title="API de Extracción y Clasificación de Párrafos Relevantes",
    description="Recibe un PDF de municipio, extrae párrafos, genera resumen general y clasifica alineación con políticas.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# 2) Función de utilidad: extraer texto del PDF (OCR o nativo)
# ------------------------------------------------------------
def extract_text_from_pdf(path: str, pagina_inicio: int, pagina_fin: int) -> str:
    """
    Abre el PDF en path y lee las páginas desde pagina_inicio hasta pagina_fin (inclusivo, 1-based).
    - Si la página tiene texto “nativo” (get_text() no vacío), lo usa.
    - Si está vacía (escaneada como imagen), hace OCR con pytesseract.
    Devuelve un string concatenado con todo el texto encontrado.
    """
    if pagina_inicio < 1 or pagina_fin < pagina_inicio:
        raise ValueError("Rango de páginas inválido: inicio debe ser >= 1 y fin >= inicio")

    texto_total = []
    with fitz.open(path) as doc:
        primero = pagina_inicio - 1
        ultimo = min(len(doc), pagina_fin) - 1

        for num_pag in range(primero, ultimo + 1):
            page = doc[num_pag]
            txt = page.get_text().strip()

            if txt:
                texto_total.append(f"--- Página {num_pag + 1} (texto nativo) ---\n{txt}\n")
            else:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_txt = pytesseract.image_to_string(img, lang="spa")
                    texto_total.append(f"--- Página {num_pag + 1} (OCR) ---\n{ocr_txt}\n")
                except Exception as e_ocr:
                    texto_total.append(f"--- Página {num_pag + 1}: ERROR OCR: {e_ocr} ---\n\n")

    return "\n".join(texto_total)

# ------------------------------------------------------------
# 3) Endpoint principal: /process_pdf/
# ------------------------------------------------------------
@app.post("/process_pdf/")
async def process_pdf(
    file: UploadFile = File(..., description="Archivo PDF de municipio"),
    municipio: str   = Form(..., description="Nombre del municipio"),
    pagina_inicio: int = Form(..., description="Página inicial (1-based)"),
    pagina_fin:    int = Form(..., description="Página final (1-based)"),
):
    """
    1) Validaciones básicas
    2) Guarda PDF en archivo temporal
    3) Extrae texto (OCR o nativo) solo en el rango pedido
    4) Llama a OpenAI para extraer párrafos relevantes
    5) Genera Resumen General a partir de los resúmenes parciales
    6) Clasifica alineación con Política de Cuidado
    7) Clasifica alineación con Política de Envejecimiento
    8) Limpia archivo temporal y devuelve JSON
    """

    # ---------------------------------
    # 3.1) Validaciones
    # ---------------------------------
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
    if pagina_inicio < 1 or pagina_fin < pagina_inicio:
        raise HTTPException(status_code=400, detail="Rango de páginas inválido")

    # ---------------------------------
    # 3.2) Guardar PDF en un temporal
    # ---------------------------------
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contenido = await file.read()
            tmp.write(contenido)
            tmp_path = tmp.name
            print(f"[{time.strftime('%X')}] PDF guardado en temporal: {tmp_path}")
    except Exception as e_save:
        raise HTTPException(status_code=500, detail=f"Error al guardar archivo temporal: {e_save}")

    # ---------------------------------
    # 3.3) Extraer texto solo del rango pedido
    # ---------------------------------
    try:
        print(f"[{time.strftime('%X')}] Iniciando extracción OCR/texto de páginas {pagina_inicio} a {pagina_fin}...")
        t0 = time.time()
        texto_municipio = extract_text_from_pdf(tmp_path, pagina_inicio, pagina_fin)
        t1 = time.time()
        print(f"[{time.strftime('%X')}] Terminó extracción de texto (duró {t1-t0:.1f} segundos).")
    except Exception as e_extract:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error al extraer texto del PDF: {e_extract}")

    # ---------------------------------
    # 3.4) Llamada a OpenAI para párrafos relevantes
    # ---------------------------------
    prompt_parrafos = f"""
Eres un asistente que extrae párrafos relevantes de documentos de planeación territorial.
Busca párrafos que contengan cualquiera de los siguientes temas (en español):
- Personas mayores
- Adulto mayor
- Personas cuidadoras
- Personas que requieren cuidado
- Cuidadores
- Sistema de cuidado
- Caracterización de personas mayores
- Caracterización de cuidadores o personas cuidadoras

Para cada párrafo que encuentres, genera un objeto JSON con estos campos exactos:
  - "pagina": número de página (1-based)
  - "parrafo": texto completo del párrafo
  - "tema": cuál de los temas detectaste
  - "resumen": breve resumen (1–2 líneas) del párrafo
  - "relevancia": uno de ["baja", "media", "alta"]

**Devuelve únicamente un ARRAY JSON (puede estar vacío) sin texto extra, ni explicaciones, ni backticks.**

Municipio: {municipio}
Rango de páginas: {pagina_inicio} a {pagina_fin}

Texto a analizar:
\"\"\"
{texto_municipio}
\"\"\"
"""

    try:
        print(f"[{time.strftime('%X')}] Enviando prompt a OpenAI para extraer párrafos relevantes...")
        t0 = time.time()
        response1 = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user", "content": prompt_parrafos}],
            temperature=0.0,
            max_tokens=2000
        )
        response_text1 = response1.choices[0].message.content.strip()
        t1 = time.time()
        print(f"[{time.strftime('%X')}] Respuesta OpenAI (párrafos) recibida (duró {t1-t0:.1f} segundos).")
    except openai.error.OpenAIError as e_openai:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error en llamada a OpenAI (párrafos): {e_openai}")

    # Limpiar delimitadores ``` si existen
    if response_text1.startswith("```") and response_text1.endswith("```"):
        lines = response_text1.splitlines()
        if len(lines) >= 3:
            response_text1 = "\n".join(lines[1:-1]).strip()

    # Intentar parsear JSON de párrafos
    try:
        resultados_parrafos = json.loads(response_text1)
    except json.JSONDecodeError:
        resultados_parrafos = {
            "error": "No se pudo parsear JSON de párrafos",
            "raw_response": response_text1
        }

    # ---------------------------------
    # 3.5) Generar Resumen General a partir de los resúmenes parciales
    # ---------------------------------
    resumen_general = None
    try:
        if isinstance(resultados_parrafos, list) and resultados_parrafos:
            res_parciales = [item.get("resumen", "").strip()
                             for item in resultados_parrafos if isinstance(item, dict)]
            bloque_para_resumen = "\n".join(f"- {r}" for r in res_parciales if r)

            prompt_resumen = f"""
A partir de estos resúmenes parciales (cada línea corresponde a un resumen breve de un párrafo):
{bloque_para_resumen}

Genera un Resumen General coherente (2–3 párrafos) que sintetice las ideas principales de todos los resúmenes anteriores.
Devuelve únicamente el texto del Resumen General, sin encabezados ni formato extra.
"""
            print(f"[{time.strftime('%X')}] Enviando prompt a OpenAI para generar Resumen General...")
            t0 = time.time()
            resp2 = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user", "content": prompt_resumen}],
                temperature=0.0,
                max_tokens=1000
            )
            resumen_general = resp2.choices[0].message.content.strip()
            t1 = time.time()
            print(f"[{time.strftime('%X')}] Resumen General recibido (duró {t1-t0:.1f} segundos).")
        else:
            resumen_general = "No hay resúmenes parciales para generar un Resumen General."
    except Exception as e_resumen:
        resumen_general = f"Error al generar Resumen General: {e_resumen}"

    # ---------------------------------
    # 3.6) Clasificar alineación contra Política de Cuidado
    # ---------------------------------
    clasificacion_cuidado = {}
    try:
        prompt_cuidado = f"""
Eres un experto en políticas públicas de cuidado. A continuación se te proporciona:
1) Texto completo de la Política Nacional de Cuidado (secciones relevantes, formato plano):
\"\"\"\n{TEXT_CUIDADO}\n\"\"\"

2) Resumen General del municipio «{municipio}»:
\"\"\"\n{resumen_general}\n\"\"\"

A. Evalúa el nivel de ALINEACIÓN entre el Resumen General y la Política Nacional de Cuidado. Clasifica en uno de ["alta", "media", "baja"].
B. Especifica con qué objetivos de la política de cuidado se alinea el Resumen General (por ejemplo: "Objetivo 1", "Objetivo 2", etc.).

Devuelve únicamente un JSON con este formato:
{{
  "alineacion": "alta|media|baja",
  "objetivos_alineados": ["Objetivo X", "Objetivo Y", …]
}}
Sin texto adicional ni explicaciones.
"""
        print(f"[{time.strftime('%X')}] Enviando prompt a OpenAI para clasificar contra Política de Cuidado...")
        t0 = time.time()
        resp3 = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user", "content": prompt_cuidado}],
            temperature=0.0,
            max_tokens=500
        )
        text3 = resp3.choices[0].message.content.strip()
        t1 = time.time()
        print(f"[{time.strftime('%X')}] Respuesta clasificación Cuidado recibida (duró {t1-t0:.1f} segundos).")

        if text3.startswith("```") and text3.endswith("```"):
            lines = text3.splitlines()
            if len(lines) >= 3:
                text3 = "\n".join(lines[1:-1]).strip()

        clasificacion_cuidado = json.loads(text3)
    except Exception as e_cu:
        clasificacion_cuidado = {"error": f"No se pudo clasificar Cuidado: {e_cu}"}

    # ---------------------------------
    # 3.7) Clasificar alineación contra Política de Envejecimiento
    # ---------------------------------
    clasificacion_envejecimiento = {}
    try:
        prompt_envejecimiento = f"""
Eres un experto en políticas de envejecimiento y vejez. A continuación:
1) Texto completo de la Política Nacional de Envejecimiento y Vejez:
\"\"\"\n{TEXT_ENVEJECIMIENTO}\n\"\"\"

2) Resumen General del municipio «{municipio}»:
\"\"\"\n{resumen_general}\n\"\"\"

A. Evalúa el nivel de ALINEACIÓN entre el Resumen General y la Política Nacional de Envejecimiento y Vejez. Clasifica en uno de ["alta", "media", "baja"].
B. Especifica con qué objetivos de la política de envejecimiento se alinea el Resumen General (por ejemplo: "Objetivo 1", "Objetivo 2", etc.).

Devuelve únicamente un JSON con este formato:
{{
  "alineacion": "alta|media|baja",
  "objetivos_alineados": ["Objetivo X", "Objetivo Y", …]
}}
Sin texto adicional ni explicaciones.
"""
        print(f"[{time.strftime('%X')}] Enviando prompt a OpenAI para clasificar contra Política de Envejecimiento...")
        t0 = time.time()
        resp4 = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user", "content": prompt_envejecimiento}],
            temperature=0.0,
            max_tokens=500
        )
        text4 = resp4.choices[0].message.content.strip()
        t1 = time.time()
        print(f"[{time.strftime('%X')}] Respuesta clasificación Envejecimiento recibida (duró {t1-t0:.1f} segundos).")

        if text4.startswith("```") and text4.endswith("```"):
            lines = text4.splitlines()
            if len(lines) >= 3:
                text4 = "\n".join(lines[1:-1]).strip()

        clasificacion_envejecimiento = json.loads(text4)
    except Exception as e_en:
        clasificacion_envejecimiento = {"error": f"No se pudo clasificar Envejecimiento: {e_en}"}

    # ---------------------------------
    # 3.8) Limpiar archivo temporal
    # ---------------------------------
    try:
        os.remove(tmp_path)
        print(f"[{time.strftime('%X')}] Archivo temporal eliminado: {tmp_path}")
    except Exception:
        pass

    # ---------------------------------
    # 3.9) Devolver la respuesta
    # ---------------------------------
    return {
        "municipio": municipio,
        "pagina_inicio": pagina_inicio,
        "pagina_fin": pagina_fin,
        "resultados_parrafos": resultados_parrafos,
        "resumen_general": resumen_general,
        "clasificacion_cuidado": clasificacion_cuidado,
        "clasificacion_envejecimiento": clasificacion_envejecimiento
    }


# ------------------------------------------------------------
# 4) Arrancar con Uvicorn si lo ejecutamos directamente
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("function_app:app", host="0.0.0.0", port=8002, reload=False, workers=1)
