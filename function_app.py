# function_app.py

import os
import time
import tempfile
import json

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

RUTA_TXT_CUIDADO       = os.path.join(BASE_DIR, "politicas_txt", "politica_cuidado.txt")
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

# En Windows, ajusta la ruta a tesseract.exe si es necesario:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(
    title="API de Extracción y Clasificación de Párrafos Relevantes",
    description="Recibe un PDF, extrae párrafos, genera resumen e indicadores de alineación con políticas.",
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
# 2) Función de utilidad: contar páginas del PDF
# ------------------------------------------------------------
def contar_paginas_pdf(path: str) -> int:
    """ Devuelve el número de páginas que tiene el PDF en disco """
    with fitz.open(path) as doc:
        total = len(doc)
    return total

# ------------------------------------------------------------
# 3) Función de utilidad: extraer texto del PDF (OCR o nativo)
# ------------------------------------------------------------
def extract_text_from_pdf(path: str, pagina_inicio: int, pagina_fin: int) -> str:
    """
    Extrae texto de cada página en el rango [pagina_inicio..pagina_fin].
    Si la página no tiene texto simple, intenta OCR.
    """
    texto_total = []
    with fitz.open(path) as doc:
        primero = pagina_inicio - 1
        ultimo  = pagina_fin - 1
        for num_pag in range(primero, ultimo + 1):
            page = doc[num_pag]
            txt = page.get_text().strip()
            if txt:
                texto_total.append(f"--- Página {num_pag+1} (texto nativo) ---\n{txt}\n")
            else:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_txt = pytesseract.image_to_string(img, lang="spa")
                    texto_total.append(f"--- Página {num_pag+1} (OCR) ---\n{ocr_txt}\n")
                except Exception as e_ocr:
                    texto_total.append(f"--- Página {num_pag+1}: ERROR OCR: {e_ocr} ---\n\n")
    return "\n".join(texto_total)

# ------------------------------------------------------------
# 4) Endpoint principal: /process_pdf/
# ------------------------------------------------------------
@app.post("/process_pdf/")
async def process_pdf(
    file: UploadFile = File(..., description="Archivo PDF de municipio"),
    municipio: str   = Form(..., description="Nombre del municipio"),
    pagina_inicio: int = Form(..., description="Página inicial (1-based)"),
    pagina_fin:    int = Form(..., description="Página final (1-based)"),
):
    # --- 4.1) Validaciones iniciales ---
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
    if pagina_inicio < 1 or pagina_fin < pagina_inicio:
        raise HTTPException(status_code=400, detail="Rango de páginas inválido")

    # Limpio posibles '\n' o espacios extra en el nombre del municipio
    municipio = municipio.strip()
    print(f"[{time.strftime('%H:%M:%S')}] Recibido PDF para municipio: '{municipio}', páginas {pagina_inicio} a {pagina_fin}")

    # --- 4.2) Guardar PDF de municipio en un temporal ---
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contenido = await file.read()
            tmp.write(contenido)
            tmp_path = tmp.name
        print(f"[{time.strftime('%H:%M:%S')}] PDF guardado en temporal: {tmp_path}")
    except Exception as e_save:
        raise HTTPException(status_code=500, detail=f"Error al guardar archivo temporal: {e_save}")

    # --- 4.3) Contar cuántas páginas tiene el PDF ---
    try:
        total_paginas = contar_paginas_pdf(tmp_path)
        print(f"[{time.strftime('%H:%M:%S')}] El PDF tiene {total_paginas} páginas.")
    except Exception as e_contar:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error al contar páginas: {e_contar}")

    # --- 4.4) Ajustar pagina_fin al número real de páginas ---
    pagina_fin_real = min(pagina_fin, total_paginas)
    print(f"[{time.strftime('%H:%M:%S')}] Ajustado pagina_fin de {pagina_fin} a {pagina_fin_real}")

    # --- 4.5) Extraer texto por lotes de 50 páginas (o menos) ---
    lotes = []
    lote_inicio = pagina_inicio
    lote_tamano = 50

    while lote_inicio <= pagina_fin_real:
        lote_fin = min(lote_inicio + lote_tamano - 1, pagina_fin_real)
        lotes.append((lote_inicio, lote_fin))
        lote_inicio = lote_fin + 1

    print(f"[{time.strftime('%H:%M:%S')}] Se van a procesar los siguientes lotes de páginas: {lotes}")

    resultados_parrafos_todos = []
    resumenes_parciales = []

    for idx, (ini, fin) in enumerate(lotes, start=1):
        print(f"[{time.strftime('%H:%M:%S')}] Lote {idx}/{len(lotes)}: Extrayendo texto de páginas {ini} a {fin} ...")
        try:
            texto_lote = extract_text_from_pdf(tmp_path, ini, fin)
            print(f"[{time.strftime('%H:%M:%S')}] Terminé extracción de texto (lote {idx}) de páginas {ini} a {fin}.")
        except Exception as e_ext:
            os.remove(tmp_path)
            raise HTTPException(status_code=500, detail=f"Error al extraer texto del PDF (lote {idx}): {e_ext}")

        # --- 4.6) Llamada a OpenAI para extraer párrafos relevantes del lote actual ---
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
  - "pagina": número de página (1-based)  ←   Nota: suma ini-1 si quieres la numeración real
  - "parrafo": texto completo del párrafo
  - "tema": cuál de los temas detectaste
  - "resumen": breve resumen (1–2 líneas) del párrafo
  - "relevancia": uno de ["baja", "media", "alta"]

**Devuelve únicamente un ARRAY JSON (puede estar vacío) sin texto extra, ni explicaciones, ni backticks.**

Municipio: {municipio}
Rango de páginas de este lote: {ini} a {fin}

Texto a analizar:
\"\"\"{texto_lote}\"\"\"
"""
        print(f"[{time.strftime('%H:%M:%S')}] Enviando prompt a OpenAI para extraer párrafos relevantes (lote {idx})...")
        try:
            response1 = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_parrafos}],
                temperature=0.0,
                max_tokens=2000
            )
            response_text1 = response1.choices[0].message.content.strip()
            print(f"[{time.strftime('%H:%M:%S')}] Respuesta OpenAI (párrafos, lote {idx}) recibida (duró ~{response1.usage.total_tokens/1000:.1f}k tokens).")
        except openai.error.OpenAIError as e_openai:
            os.remove(tmp_path)
            raise HTTPException(status_code=500, detail=f"Error en llamada a OpenAI (párrafos, lote {idx}): {e_openai}")

        # Eliminar posibles backticks
        if response_text1.startswith("```") and response_text1.endswith("```"):
            lines = response_text1.splitlines()
            if len(lines) >= 3:
                response_text1 = "\n".join(lines[1:-1]).strip()

        # Intentar parsear JSON de párrafos
        try:
            lista_parrafos_lote = json.loads(response_text1)
        except json.JSONDecodeError:
            lista_parrafos_lote = []
            print(f"[{time.strftime('%H:%M:%S')}] ¡Advertencia! No se pudo parsear JSON de párrafos en lote {idx}. Raw:\n{response_text1[:300]}...")

        # Ajustar número de página real por cada párrafo
        for par in lista_parrafos_lote:
            # Si el asistente devolvió "pagina" relativa al lote (ej: 1..(fin-ini+1)), ajustamos así:
            pag_rel = par.get("pagina", None)
            if isinstance(pag_rel, int):
                par["pagina"] = pag_rel + (ini - 1)
            resultados_parrafos_todos.append(par)
            resumen_par = par.get("resumen", "").strip()
            if resumen_par:
                resumenes_parciales.append(resumen_par)

    # 4.7) Generar Resumen General a partir de todos los resúmenes parciales
    resumen_general = ""
    if resumenes_parciales:
        bloque_resumenes = "\n".join(f"- {r}" for r in resumenes_parciales)
        prompt_resumen = f"""
A partir de estos resúmenes parciales (cada línea corresponde a un breve resumen de un párrafo):
{bloque_resumenes}

Genera un Resumen General coherente (2–3 párrafos) que sintetice las ideas principales de todos los resúmenes anteriores.
Devuelve únicamente el texto del Resumen General, sin encabezados ni formato extra.
"""
        print(f"[{time.strftime('%H:%M:%S')}] Enviando prompt a OpenAI para generar Resumen General...")
        try:
            resp2 = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_resumen}],
                temperature=0.0,
                max_tokens=1000
            )
            resumen_general = resp2.choices[0].message.content.strip()
            print(f"[{time.strftime('%H:%M:%S')}] Resumen General recibido.")
        except Exception as e_resumen:
            resumen_general = f"Error al generar Resumen General: {e_resumen}"
            print(f"[{time.strftime('%H:%M:%S')}] ¡Error generando Resumen General!: {e_resumen}")
    else:
        resumen_general = "No hay resúmenes parciales para generar un Resumen General."
        print(f"[{time.strftime('%H:%M:%S')}] No había párrafos para resumir. El Resumen General quedó vacío.")

    # 4.8) Clasificar alineación contra Política de Cuidado
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
        print(f"[{time.strftime('%H:%M:%S')}] Enviando prompt a OpenAI para clasificar alineación con Política de Cuidado...")
        resp3 = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_cuidado}],
            temperature=0.0,
            max_tokens=500
        )
        text3 = resp3.choices[0].message.content.strip()
        if text3.startswith("```") and text3.endswith("```"):
            lines = text3.splitlines()
            if len(lines) >= 3:
                text3 = "\n".join(lines[1:-1]).strip()
        clasificacion_cuidado = json.loads(text3)
        print(f"[{time.strftime('%H:%M:%S')}] Clasificación Cuidado recibida.")
    except Exception as e_cu:
        clasificacion_cuidado = {"error": f"No se pudo clasificar Cuidado: {e_cu}"}
        print(f"[{time.strftime('%H:%M:%S')}] ¡Error clasificando Cuidado!: {e_cu}")

    # 4.9) Clasificar alineación contra Política de Envejecimiento
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
        print(f"[{time.strftime('%H:%M:%S')}] Enviando prompt a OpenAI para clasificar alineación con Política de Envejecimiento...")
        resp4 = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_envejecimiento}],
            temperature=0.0,
            max_tokens=500
        )
        text4 = resp4.choices[0].message.content.strip()
        if text4.startswith("```") and text4.endswith("```"):
            lines = text4.splitlines()
            if len(lines) >= 3:
                text4 = "\n".join(lines[1:-1]).strip()
        clasificacion_envejecimiento = json.loads(text4)
        print(f"[{time.strftime('%H:%M:%S')}] Clasificación Envejecimiento recibida.")
    except Exception as e_en:
        clasificacion_envejecimiento = {"error": f"No se pudo clasificar Envejecimiento: {e_en}"}
        print(f"[{time.strftime('%H:%M:%S')}] ¡Error clasificando Envejecimiento!: {e_en}")

    # 4.10) Limpiar archivo temporal del municipio
    try:
        os.remove(tmp_path)
        print(f"[{time.strftime('%H:%M:%S')}] Archivo temporal eliminado: {tmp_path}")
    except Exception:
        pass

    # 4.11) Devolver la respuesta completa
    return {
        "municipio": municipio,
        "pagina_inicio": pagina_inicio,
        "pagina_fin": pagina_fin_real,
        "resultados_parrafos": resultados_parrafos_todos,
        "resumen_general": resumen_general,
        "clasificacion_cuidado": clasificacion_cuidado,
        "clasificacion_envejecimiento": clasificacion_envejecimiento
    }

# ------------------------------------------------------------
# 5) Arrancar con Uvicorn si se ejecuta directamente
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("function_app:app", host="0.0.0.0", port=8002, reload=True)
