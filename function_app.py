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

# En Windows, ajuste la ruta de tesseract si es necesario:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
# 2) Función auxiliar: Extrae texto nativo y OCR de un rango de páginas
# ────────────────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(path: str, pagina_inicio: int, pagina_fin: int) -> str:
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
                texto_total.append(f"--- Página {num_pag+1} (texto nativo) ---\n{txt}\n")
            else:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_txt = pytesseract.image_to_string(img, lang="spa").strip()
                    texto_total.append(f"--- Página {num_pag+1} (OCR) ---\n{ocr_txt}\n")
                except Exception as e_ocr:
                    texto_total.append(f"--- Página {num_pag+1}: ERROR OCR: {e_ocr} ---\n\n")
    return "\n".join(texto_total)

# Para timestamp en prints
now = lambda: datetime.datetime.now().strftime("%H:%M:%S")

# ────────────────────────────────────────────────────────────────────────────────
# 3) Endpoint principal: /process_pdf/
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/process_pdf/")
async def process_pdf(
    file: UploadFile = File(..., description="Archivo PDF"),
    municipio: str = Form(..., description="Nombre del municipio"),
):
    # 3.1) Validaciones y guardado temporal del PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contenido = await file.read()
            tmp.write(contenido)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar PDF: {e}")

    # 3.2) Obtener total de páginas
    try:
        with fitz.open(tmp_path) as doc:
            total_paginas = len(doc)
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error al abrir PDF: {e}")

    print(f"[{now()}] PDF guardado en temporal: {tmp_path}")
    print(f"[{now()}] El PDF tiene {total_paginas} páginas.")

    # 3.3) Procesar en lotes para no exceder límite de tokens
    lista_parrafos = []
    lote_size = 20  # Número de páginas por lote

    for inicio in range(1, total_paginas + 1, lote_size):
        fin = min(inicio + lote_size - 1, total_paginas)
        print(f"[{now()}] → Procesando lote páginas {inicio} a {fin}…")

        # Extraer texto (nativo u OCR) en el rango
        start_lote = datetime.datetime.now()
        try:
            texto_lote = extract_text_from_pdf(tmp_path, inicio, fin)
        except Exception as e:
            os.remove(tmp_path)
            raise HTTPException(
                status_code=500,
                detail=f"Error extrayendo texto (páginas {inicio}-{fin}): {e}"
            )
        elapsed_lote = (datetime.datetime.now() - start_lote).total_seconds()
        print(f"[{now()}] ---- Texto extraído (Lote {inicio}-{fin}) en {round(elapsed_lote, 1)}s.")

        # Debug: mostrar primeros caracteres
        fragmento_debug = texto_lote[:200] if texto_lote else ""
        print(f"[DEBUG {now()}] Fragmento (páginas {inicio}-{fin}): {fragmento_debug!r}\n")

        # Cortar a un máximo de 20 000 caracteres para el prompt
        texto_para_prompt = texto_lote[:20000]

        # Construir prompt más compacto para extracción de párrafos
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

Texto (páginas {inicio}-{fin}, recortado a 20000 caracteres):
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
            os.remove(tmp_path)
            raise HTTPException(
                status_code=500,
                detail=f"Error OpenAI (párrafos, páginas {inicio}-{fin}): {e_openai}"
            )
        elapsed_call = (datetime.datetime.now() - start_call).total_seconds()
        print(f"[{now()}] ---- Respuesta OpenAI (párrafos) recibida en {round(elapsed_call, 1)}s.")

        # Quitar delimitadores ``` si los hubiera
        if response_text1.startswith("```") and response_text1.endswith("```"):
            lines = response_text1.splitlines()
            if len(lines) >= 3:
                response_text1 = "\n".join(lines[1:-1]).strip()

        # Intentar parsear JSON; si falla, registrar error en un objeto
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

        # Acumular resultados
        lista_parrafos.extend(resultados_parciales)

    # 3.4) Generar Resumen General a partir de los resúmenes parciales
    print(f"[{now()}] Generando Resumen General a partir de {len(lista_parrafos)} ítems…")
    resumen_general = ""
    try:
        # Extraer únicamente los “resumen” válidos de cada elemento
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

    # 3.5) Clasificar alineación contra Política de Cuidado
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

    # 3.6) Clasificar alineación contra Política de Envejecimiento y Vejez
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

    # 3.7) Limpiar archivo temporal
    os.remove(tmp_path)
    print(f"[{now()}] Archivo temporal eliminado: {tmp_path}")

    # 3.8) Respuesta final
    return {
        "municipio": municipio,
        "total_paginas": total_paginas,
        "resultados_parrafos": lista_parrafos,
        "resumen_general": resumen_general,
        "clasificacion_cuidado": clasificacion_cuidado,
        "clasificacion_envejecimiento": clasificacion_envejecimiento,
    }


# ────────────────────────────────────────────────────────────────────────────────
# 4) Ejecutar con Uvicorn si se llama directamente
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("function_app:app", host="0.0.0.0", port=8002, workers=1)
