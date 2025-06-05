# function_app.py

import os
import json
import tempfile
import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pytesseract
from PIL import Image
import fitz   # PyMuPDF

import openai

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

# Si usas Windows y Tesseract, ajusta la ruta:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
    Extrae el texto de las páginas [pagina_inicio .. pagina_fin] (1-based) de un PDF.
    Si la página no tiene texto nativo, hace OCR con pytesseract.
    Devuelve un gran string concatenado (separado por marcadores de página).
    """
    if pagina_inicio < 1 or pagina_fin < pagina_inicio:
        raise ValueError("Rango de páginas inválido")

    texto_total = []
    with fitz.open(path) as doc:
        total = len(doc)
        primero = pagina_inicio - 1
        ultimo = min(total, pagina_fin) - 1

        for num_pag in range(primero, ultimo + 1):
            page = doc[num_pag]
            txt = page.get_text().strip()
            if txt:
                texto_total.append(f"--- Página {num_pag + 1} (texto nativo) ---\n{txt}\n")
            else:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_txt = pytesseract.image_to_string(img, lang="spa").strip()
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
    municipio: str = Form(..., description="Nombre del municipio"),
):
    """
    Recibe un PDF completo y un nombre de municipio.
    Divide internamente el PDF en lotes de 50 páginas, extrae texto y párrafos relevantes,
    luego genera resumen general, clasifica alineación con dos políticas y retorna JSON completo.
    """

    # 3.1) Validaciones básicas
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

    # 3.2) Guardar PDF de municipio en archivo temporal
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contenido = await file.read()
            tmp.write(contenido)
            tmp_path = tmp.name
    except Exception as e_save:
        raise HTTPException(status_code=500, detail=f"Error al guardar archivo temporal: {e_save}")

    # 3.3) Abrir PDF para conocer número de páginas
    try:
        with fitz.open(tmp_path) as doc:
            total_paginas = len(doc)
    except Exception as e_open:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error al abrir PDF: {e_open}")

    now = lambda: datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now()}] PDF guardado en temporal: {tmp_path}")
    print(f"[{now()}] El PDF tiene {total_paginas} páginas.")

    # 3.4) Preparar lista para juntar todos los párrafos relevantes
    lista_parrafos = []

    # Procesar en lotes de 50 páginas
    lote_size = 50
    for inicio in range(1, total_paginas + 1, lote_size):
        fin = min(inicio + lote_size - 1, total_paginas)
        print(f"[{now()}] → Procesando lote páginas {inicio} a {fin}…")

        # 3.4.1) Extraer texto (OCR o nativo) de este lote
        start_lote = datetime.datetime.now()
        try:
            texto_lote = extract_text_from_pdf(tmp_path, inicio, fin)
        except Exception as e_extract:
            os.remove(tmp_path)
            raise HTTPException(
                status_code=500,
                detail=f"Error al extraer texto del PDF (páginas {inicio}-{fin}): {e_extract}"
            )
        elapsed_lote = (datetime.datetime.now() - start_lote).total_seconds()
        print(f"[{now()}] ---- Texto extraído (Lote {inicio}-{fin}) en {round(elapsed_lote, 1)}s.")

        # 3.4.2) Llamada a OpenAI para extraer párrafos relevantes de este bloque
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
  - "pagina": número de página (1-based, relativo al documento completo)
  - "parrafo": texto completo del párrafo
  - "tema": cuál de los temas detectaste
  - "resumen": breve resumen (1–2 líneas) del párrafo
  - "relevancia": uno de ["baja", "media", "alta"]

**Devuelve únicamente un ARRAY JSON (puede estar vacío) sin texto extra, ni explicaciones, ni backticks.**

Municipio: {municipio}
Rango de páginas actuales: {inicio} a {fin}

Texto a analizar:
\"\"\"{texto_lote}\"\"\"
"""
        print(f"[{now()}] Enviando prompt a OpenAI para extraer párrafos (Páginas {inicio}-{fin})…")
        start_call = datetime.datetime.now()
        try:
            response1 = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_parrafos}],
                temperature=0.0,
                max_tokens=2000,
            )
            response_text1 = response1.choices[0].message.content.strip()
        except openai.error.OpenAIError as e_openai:
            os.remove(tmp_path)
            raise HTTPException(
                status_code=500,
                detail=f"Error en llamada a OpenAI (párrafos, páginas {inicio}-{fin}): {e_openai}"
            )
        elapsed_call = (datetime.datetime.now() - start_call).total_seconds()
        print(f"[{now()}] ---- Respuesta OpenAI (párrafos) recibida en {round(elapsed_call, 1)}s.")

        # Limpiar triple backticks si aparecen
        if response_text1.startswith("```") and response_text1.endswith("```"):
            lines = response_text1.splitlines()
            if len(lines) >= 3:
                response_text1 = "\n".join(lines[1:-1]).strip()

        # Intentar parsear JSON devuelto
        try:
            resultados_parciales = json.loads(response_text1)
        except json.JSONDecodeError:
            # Si no es JSON válido, guardamos un error como dict dentro de la lista
            resultados_parciales = [{
                "pagina": None,
                "parrafo": None,
                "tema": None,
                "resumen": None,
                "relevancia": None,
                "error": "No se pudo parsear JSON",
                "raw_response": response_text1
            }]

        # Agregar a la lista global
        lista_parrafos.extend(resultados_parciales)

    # 3.5) Generar Resumen General a partir de los resúmenes parciales extraídos
    print(f"[{now()}] Generando Resumen General a partir de {len(lista_parrafos)} resúmenes parciales…")
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
A partir de estos resúmenes parciales (cada línea corresponde a un resumen breve de un párrafo):
{bloque_para_resumen}

Genera un Resumen General coherente (2–3 párrafos) que sintetice las ideas principales de todos los resúmenes anteriores.
Devuelve únicamente el texto del Resumen General, sin encabezados ni formato extra.
"""
            start_resumen = datetime.datetime.now()
            resp2 = openai.chat.completions.create(
                model="gpt-4o-mini",
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

    # 3.6) Clasificar alineación contra Política de Cuidado
    print(f"[{now()}] Clasificando alineación contra Política de Cuidado…")
    clasificacion_cuidado = {}
    try:
        prompt_cuidado = f"""
Eres un experto en políticas públicas de cuidado. A continuación se te proporciona:
1) Texto completo de la Política Nacional de Cuidado (secciones relevantes, formato plano):
\"\"\"\n{TEXT_CUIDADO}\n\"\"\"

2) Resumen General del municipio «{municipio}»:
\"\"\"\n{resumen_general}\n\"\"\"

A. Evalúa el nivel de ALINEACIÓN entre el Resumen General y la Política Nacional de Cuidado.
Clasifica en uno de ["alta", "media", "baja"].
B. Especifica con qué objetivos de la política de cuidado se alinea el Resumen General
   (por ejemplo: "Objetivo 1", "Objetivo 2", …).

Devuelve únicamente un JSON con este formato:
{{
  "alineacion": "alta|media|baja",
  "objetivos_alineados": ["Objetivo X", "Objetivo Y", …]
}}
Sin texto adicional ni explicaciones.
"""
        start_cuidado = datetime.datetime.now()
        resp3 = openai.chat.completions.create(
            model="gpt-4o-mini",
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

    # 3.7) Clasificar alineación contra Política de Envejecimiento
    print(f"[{now()}] Clasificando alineación contra Política de Envejecimiento…")
    clasificacion_envejecimiento = {}
    try:
        prompt_envejecimiento = f"""
Eres un experto en políticas de envejecimiento y vejez. A continuación:
1) Texto completo de la Política Nacional de Envejecimiento y Vejez:
\"\"\"\n{TEXT_ENVEJECIMIENTO}\n\"\"\"

2) Resumen General del municipio «{municipio}»:
\"\"\"\n{resumen_general}\n\"\"\"

A. Evalúa el nivel de ALINEACIÓN entre el Resumen General y la Política Nacional de Envejecimiento y Vejez.
   Clasifica en uno de ["alta", "media", "baja"].
B. Especifica con qué objetivos de la política de envejecimiento se alinea el Resumen General
   (por ejemplo: "Objetivo 1", "Objetivo 2", …).

Devuelve únicamente un JSON con este formato:
{{
  "alineacion": "alta|media|baja",
  "objetivos_alineados": ["Objetivo X", "Objetivo Y", …]
}}
Sin texto adicional ni explicaciones.
"""
        start_enve = datetime.datetime.now()
        resp4 = openai.chat.completions.create(
            model="gpt-4o-mini",
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

    # 3.8) Limpiar archivo temporal del municipio
    os.remove(tmp_path)
    print(f"[{now()}] Archivo temporal eliminado: {tmp_path}")

    # 3.9) Devolver la respuesta completa
    return {
        "municipio": municipio,
        "total_paginas": total_paginas,
        "resultados_parrafos": lista_parrafos,
        "resumen_general": resumen_general,
        "clasificacion_cuidado": clasificacion_cuidado,
        "clasificacion_envejecimiento": clasificacion_envejecimiento,
    }


# ------------------------------------------------------------
# 4) Arrancar con Uvicorn si lo ejecutamos directamente
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("function_app:app", host="0.0.0.0", port=8002, workers=1)
