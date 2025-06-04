# function_app.py

import os
import time
import json
import tempfile
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

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

# Si usas Windows, ajusta la ruta de tesseract:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(
    title="API de Extracción y Clasificación de Párrafos Relevantes",
    description="Recibe un PDF de municipio, extrae párrafos y genera resumen y clasificación.",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# 2) Función de utilidad: contar cuántas páginas tiene un PDF
# ------------------------------------------------------------
def contar_paginas_pdf(path: str) -> int:
    """
    Abre el PDF en la ruta 'path' y devuelve su cantidad total de páginas.
    """
    try:
        with fitz.open(path) as doc:
            total = len(doc)
        return total
    except Exception as e:
        raise RuntimeError(f"Error al abrir el PDF para contar páginas: {e}")

# ------------------------------------------------------------
# 3) Función de utilidad: extraer texto de páginas dadas
# ------------------------------------------------------------
def extract_text_from_pdf(
    path: str,
    pagina_inicio: int,
    pagina_fin: int
) -> str:
    """
    Extrae texto nativo de PDF en el rango [pagina_inicio, pagina_fin].
    Si la página está vacía, aplica OCR con pytesseract.
    """
    texto_total: List[str] = []
    with fitz.open(path) as doc:
        primero = max(pagina_inicio - 1, 0)
        ultimo = min(len(doc), pagina_fin) - 1
        for num_pag in range(primero, ultimo + 1):
            page = doc[num_pag]
            txt = page.get_text().strip()
            if txt:
                texto_total.append(f"--- Página {num_pag+1} (texto nativo) ---\n{txt}\n")
            else:
                # Si no hay texto nativo, aplicamos OCR
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_txt = pytesseract.image_to_string(img, lang="spa")
                    texto_total.append(f"--- Página {num_pag+1} (OCR) ---\n{ocr_txt}\n")
                except Exception as e_ocr:
                    texto_total.append(f"--- Página {num_pag+1}: ERROR OCR: {e_ocr} ---\n\n")
    return "\n".join(texto_total)

# ------------------------------------------------------------
# 4) Función de utilidad: llamar a OpenAI para extraer párrafos
# ------------------------------------------------------------
def llamar_openai_para_parrafos(
    texto_a_analizar: str,
    municipio: str,
    pagina_inicio: int,
    pagina_fin: int
) -> List[Dict]:
    """
    Manda el prompt a OpenAI (gpt-4o-mini) para extraer párrafos relevantes
    dentro de texto_a_analizar. Devuelve LISTA de objetos con keys:
      - "pagina", "parrafo", "tema", "resumen", "relevancia"
    Si falla el parseo JSON, devuelve lista vacía.
    """
    prompt = f"""
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
{texto_a_analizar}
\"\"\"
"""
    print(f"[{time.strftime('%H:%M:%S')}] Enviando prompt a OpenAI para extraer párrafos (páginas {pagina_inicio}-{pagina_fin})…")
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2500  # suficiente para devolver JSON de párrafos
        )
        respuesta_texto = resp.choices[0].message.content.strip()
        # Limpiar triple backticks si viniera envuelto en ```json ... ```
        if respuesta_texto.startswith("```") and respuesta_texto.endswith("```"):
            lines = respuesta_texto.splitlines()
            if len(lines) >= 3:
                respuesta_texto = "\n".join(lines[1:-1]).strip()

        resultados = json.loads(respuesta_texto)
        print(f"[{time.strftime('%H:%M:%S')}] Respuesta párrafos recibida (páginas {pagina_inicio}-{pagina_fin}).")
        return resultados

    except openai.error.OpenAIError as oe:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR en ChatCompletion (párrafos): {oe}")
        return []
    except json.JSONDecodeError as je:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR al parsear JSON de párrafos: {je}\nRaw:\n{respuesta_texto}")
        return []

# ------------------------------------------------------------
# 5) Función de utilidad: llamar a OpenAI para generar resumen general
# ------------------------------------------------------------
def llamar_openai_para_resumen_general(resumenes_parciales: List[str]) -> str:
    """
    A partir de una lista de resúmenes parciales (1–2 líneas por párrafo),
    genera un resumen general (2–3 párrafos).
    Si no hay resúmenes parciales, devuelve un texto indicando que no hay.
    """
    if not resumenes_parciales:
        return "No hay resúmenes parciales para generar un Resumen General."

    bloque = "\n".join(f"- {r}" for r in resumenes_parciales if r)
    prompt = f"""
A partir de estos resúmenes parciales (cada línea corresponde a un resumen breve de un párrafo):
{bloque}

Genera un Resumen General coherente (2–3 párrafos) que sintetice las ideas principales de todos los resúmenes anteriores.
Devuelve únicamente el texto del Resumen General, sin encabezados ni formato extra.
"""
    print(f"[{time.strftime('%H:%M:%S')}] Enviando prompt a OpenAI para generar Resumen General…")
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500
        )
        resumen = resp.choices[0].message.content.strip()
        print(f"[{time.strftime('%H:%M:%S')}] Resumen General recibido.")
        return resumen
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR en generar Resumen General: {e}")
        return f"Error al generar Resumen General: {e}"

# ------------------------------------------------------------
# 6) Función de utilidad: clasificar alineación con políticas
# ------------------------------------------------------------
def llamar_openai_para_clasificacion(
    texto_politica: str,
    nombre_politica: str,
    resumen_general: str
) -> Dict:
    """
    Dado el texto completo de una política (texto_politica),
    y el resumen general (resumen_general), envía un prompt a OpenAI
    (model: gpt-3.5-turbo) para evaluar la alineación y extraer objetivos.
    Devuelve un diccionario con { "alineacion": "alta|media|baja", "objetivos_alineados": [ ... ] }.
    """
    # Si el texto de la política es demasiado largo, podríamos recortarlo a los primeros N caracteres:
    if len(texto_politica) > 4000:
        texto_cortado = texto_politica[:4000] + "\n…(texto recortado para clasificación)…"
    else:
        texto_cortado = texto_politica

    prompt = f"""
Eres un experto en políticas públicas de {nombre_politica}. A continuación se te proporciona:
1) Fragmento relevante de la política (formato plano, sin formato adicional):
\"\"\"\n{texto_cortado}\n\"\"\"

2) Resumen General del municipio:
\"\"\"\n{resumen_general}\n\"\"\"

A. Evalúa el nivel de ALINEACIÓN entre el Resumen General y la política. Clasifica en uno de ["alta", "media", "baja"].
B. Especifica con qué objetivos de la política se alinea (por ejemplo: "Objetivo 1", "Objetivo 2", etc.).

Devuelve únicamente un JSON con este formato:
{{ 
  "alineacion": "alta|media|baja",
  "objetivos_alineados": ["Objetivo X", "Objetivo Y", …]
}}
Sin texto adicional ni explicaciones.
"""

    print(f"[{time.strftime('%H:%M:%S')}] Enviando prompt a OpenAI para clasificar {nombre_politica}…")
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800
        )
        text = resp.choices[0].message.content.strip()
        # Limpiar backticks
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        resultado = json.loads(text)
        print(f"[{time.strftime('%H:%M:%S')}] Clasificación {nombre_politica} recibida.")
        return resultado
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR al clasificar {nombre_politica}: {e}")
        return {"alineacion": "error", "objetivos_alineados": [], "error": str(e)}

# ------------------------------------------------------------
# 7) Endpoint principal: /process_pdf/
# ------------------------------------------------------------
@app.post("/process_pdf/")
async def process_pdf(
    file: UploadFile = File(..., description="Archivo PDF de municipio"),
    municipio: str = Form(..., description="Nombre del municipio"),
    pagina_inicio: int = Form(..., description="Página inicial (1-based)"),
    pagina_fin: int = Form(..., description="Página final (1-based)")
):
    # 1) Validaciones básicas
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
    if pagina_inicio < 1 or pagina_fin < pagina_inicio:
        raise HTTPException(status_code=400, detail="Rango de páginas inválido")

    # 2) Guardar PDF en un archivo temporal
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contenido = await file.read()
            tmp.write(contenido)
            tmp_path = tmp.name
        print(f"[{time.strftime('%H:%M:%S')}] PDF guardado en temporal: {tmp_path}")
    except Exception as e_save:
        raise HTTPException(status_code=500, detail=f"Error al guardar archivo temporal: {e_save}")

    # 3) Contar cuántas páginas tiene el PDF
    try:
        total_paginas = contar_paginas_pdf(tmp_path)
        print(f"[{time.strftime('%H:%M:%S')}] El PDF tiene {total_paginas} páginas.")
    except Exception as e_contar:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error al contar páginas: {e_contar}")

    # 4) Ajustar el rango final si supera el total
    pagina_fin_real = min(pagina_fin, total_paginas)

    # 5) Log
    print(f"[{time.strftime('%H:%M:%S')}] Procesando municipio '{municipio}' de página {pagina_inicio} a {pagina_fin_real}...")

    # 6) Dividir en lotes de, por ejemplo, 50 páginas cada uno
    LOTE_SIZE = 50
    rangos_lotes = []
    pag_act = pagina_inicio
    while pag_act <= pagina_fin_real:
        fin_lote = min(pag_act + LOTE_SIZE - 1, pagina_fin_real)
        rangos_lotes.append((pag_act, fin_lote))
        pag_act = fin_lote + 1

    resultados_todos_lotes: List[Dict] = []

    # 7) Iterar lote por lote
    for (ini, fin) in rangos_lotes:
        print(f"[{time.strftime('%H:%M:%S')}] → Procesando lote páginas {ini} a {fin}…")

        # 7.1) Extraer texto de ese lote
        t0 = time.time()
        try:
            texto_parcial = extract_text_from_pdf(tmp_path, ini, fin)
            dur = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] ---- Texto extraído (lote {ini}-{fin}) en {dur:.1f}s.")
        except Exception as e_ext:
            os.remove(tmp_path)
            raise HTTPException(status_code=500, detail=f"Error al extraer texto (lote {ini}-{fin}): {e_ext}")

        # 7.2) Llamar a OpenAI para extraer párrafos
        resultados_parciales = llamar_openai_para_parrafos(texto_parcial, municipio, ini, fin)
        # Nos quedamos sólo con la lista, en caso de que venga None o string, forzamos lista vacía
        if not isinstance(resultados_parciales, list):
            resultados_parciales = []

        # 7.3) Agregar esos resultados parciales al arreglo global
        resultados_todos_lotes.extend(resultados_parciales)

    # 8) Ya acumulamos todos los párrafos de todos los lotes, por tanto ahora generamos el resumen general
    lista_resumenes = [item.get("resumen", "").strip() for item in resultados_todos_lotes if isinstance(item, dict)]
    resumen_general = llamar_openai_para_resumen_general(lista_resumenes)

    # 9) Clasificación con Política de Cuidado y Envejecimiento (usamos gpt-3.5-turbo)
    clasificacion_cuidado = llamar_openai_para_clasificacion(
        texto_politica=TEXT_CUIDADO,
        nombre_politica="Cuidado",
        resumen_general=resumen_general
    )
    clasificacion_envejecimiento = llamar_openai_para_clasificacion(
        texto_politica=TEXT_ENVEJECIMIENTO,
        nombre_politica="Envejecimiento",
        resumen_general=resumen_general
    )

    # 10) Borrar el temporal
    try:
        os.remove(tmp_path)
        print(f"[{time.strftime('%H:%M:%S')}] Archivo temporal eliminado: {tmp_path}")
    except:
        pass

    # 11) Construir respuesta final
    return {
        "municipio": municipio.strip(),
        "pagina_inicio": pagina_inicio,
        "pagina_fin": pagina_fin_real,
        "resultados_parrafos": resultados_todos_lotes,
        "resumen_general": resumen_general,
        "clasificacion_cuidado": clasificacion_cuidado,
        "clasificacion_envejecimiento": clasificacion_envejecimiento,
        # Si necesitas el ID del archivo en Drive, agrégalo (asumimos que se envió en el JSON inicial)
        # "id_archivo_drive": id_archivo_drive  
    }


# ------------------------------------------------------------
# 12) Ejecutar con Uvicorn al correr directamente
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("function_app:app", host="0.0.0.0", port=8002, reload=True)
