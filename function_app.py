# function_app.py

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import tempfile
import os
import openai
import json

# ------------------------------------------------------------
# 1) Configuraciones iniciales
# ------------------------------------------------------------

# (1.1) Indica aquí tu API Key de OpenAI, o asegúrate de exportarla como variable de entorno:
#      export OPENAI_API_KEY="sk-XXXXXX"
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("No se encontró la variable de entorno OPENAI_API_KEY")

# (1.2) Ruta de tesseract en Linux. En Ubuntu suele estar instalado en /usr/bin/tesseract
#       Si lo instalaste con apt-get, deja esta línea. Si lo compilas en otra ruta, ajústala.
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# (1.3) Crea la aplicación FastAPI
app = FastAPI(
    title="API de Extracción de Párrafos Relevantes",
    description="Recibe un PDF, un municipio y un rango de páginas; devuelve JSON con párrafos relevantes.",
    version="1.0",
)

# (1.4) Si vas a llamar desde una interfaz web o desde n8n en otro host, habilita CORS mínimamente:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Cambia esto a tu dominio/n8n si lo deseas más restrictivo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# 2) Función de utilidad: extraer texto (OCR o nativo) sólo del rango de páginas solicitado
# ------------------------------------------------------------
def extract_text_from_pdf(path: str, pagina_inicio: int, pagina_fin: int) -> str:
    """
    Abre el PDF en `path` y lee las páginas desde pagina_inicio hasta pagina_fin (incluyentes, 1‐based).
    - Si la página tiene texto “nativo” (get_text() no vacío), lo toma.
    - Si está vacía (escaneada como imagen), hace OCR con pytesseract.
    Devuelve un string grande que concatena todas esas páginas con saltos de línea.
    """
    if pagina_inicio < 1 or pagina_fin < pagina_inicio:
        raise ValueError("Rango de páginas inválido: inicio debe ser >= 1 y fin >= inicio")

    texto_total = []
    with fitz.open(path) as doc:
        # Ajustamos a 1‐based vs 0‐based internamente
        primero = pagina_inicio - 1
        ultimo  = min(len(doc), pagina_fin) - 1

        for num_pag in range(primero, ultimo + 1):
            page = doc[num_pag]

            # (a) Intentamos extraer texto nativo
            txt = page.get_text().strip()
            if txt:
                texto_total.append(f"--- Página {num_pag + 1} (texto nativo) ---\n{txt}\n")
            else:
                # (b) Si no hay texto, renderizamos a pixmap y hacemos OCR
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # escala 2x para mejor OCR
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_txt = pytesseract.image_to_string(img, lang="spa")
                    texto_total.append(f"--- Página {num_pag + 1} (OCR) ---\n{ocr_txt}\n")
                except Exception as e_ocr:
                    # Si falla el OCR, registramos un mensaje y continuamos
                    texto_total.append(f"--- Página {num_pag + 1}: ERROR OCR: {e_ocr} ---\n\n")

    return "\n".join(texto_total)


# ------------------------------------------------------------
# 3) Endpoint principal: /process_pdf/
# ------------------------------------------------------------
@app.post("/process_pdf/")
async def process_pdf(
    file: UploadFile = File(..., description="Archivo PDF a procesar"),
    municipio: str   = Form(..., description="Nombre del municipio"),
    pagina_inicio: int = Form(..., description="Página inicial (1-based)"),
    pagina_fin:    int = Form(..., description="Página final (1-based)"),
):
    """
    1) Se recibe un PDF (binario), un municipio y un rango de páginas
    2) Se guarda el PDF en un archivo temporal
    3) Se invoca extract_text_from_pdf(...) para extraer sólo el rango pedido
    4) Se construye el prompt para OpenAI pidiendo JSON puro con párrafos relevantes
    5) Llamada a openai.ChatCompletion.create(…)
    6) Se limpia la respuesta de triple backticks (```json …```)
    7) Se intenta parsear JSON; si falla, devolvemos raw_response
    8) Borrar archivo temporal y retornar respuesta
    """

    # -------------------------------
    # 3.1) Validaciones básicas
    # -------------------------------
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

    if pagina_inicio < 1 or pagina_fin < pagina_inicio:
        raise HTTPException(status_code=400, detail="Rango de páginas inválido")

    # -------------------------------
    # 3.2) Guardar el PDF en un temporal
    # -------------------------------
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contenido = await file.read()
            tmp.write(contenido)
            tmp_path = tmp.name
    except Exception as e_save:
        raise HTTPException(status_code=500, detail=f"Error al guardar archivo temporal: {e_save}")

    # -------------------------------
    # 3.3) Extraer texto sólo de las páginas pedidas
    # -------------------------------
    try:
        texto = extract_text_from_pdf(tmp_path, pagina_inicio, pagina_fin)
    except Exception as e_extract:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error al extraer texto del PDF: {e_extract}")

    # -------------------------------
    # 3.4) Construir el prompt para OpenAI
    #      (Le pedimos únicamente JSON en formato de array)
    # -------------------------------
    prompt = f"""
Eres un asistente que extrae párrafos relevantes de documentos de planeación territorial.
Busca párrafos que contengan cualquiera de los siguientes temas (en español):

- Enfoque diferencial
- Personas mayores
- Adulto mayor
- Cuidadores
- Sistemas de cuidado
- Caracterización poblacional

Para cada párrafo que encuentres, genera un objeto JSON con exactamente estos campos:
  - "pagina": número de página (1-based) donde aparece el párrafo
  - "parrafo": texto completo del párrafo
  - "tema": cuál de los temas anteriores se detectó en ese párrafo
  - "resumen": breve resumen (1–2 líneas) de la idea principal del párrafo
  - "relevancia": uno de ["baja", "media", "alta"] según cuán importante sea el párrafo

**Devuelve únicamente un ARRAY JSON (puede estar vacío) con todos esos objetos.**
**NO agregues texto extra, encabezados, ni comentarios, y NO uses triple backticks (```)...**

Municipio: {municipio}
Rango de páginas: {pagina_inicio} a {pagina_fin}

Texto a analizar:
\"\"\"
{texto}
\"\"\"
"""

    # -------------------------------
    # 3.5) Llamada a la API de OpenAI
    # -------------------------------
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        response_text = response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e_openai:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error en llamada a OpenAI: {e_openai}")

    # -------------------------------
    # 3.6) Limpiar los delimitadores ``` si existen
    # -------------------------------
    if response_text.startswith("```") and response_text.endswith("```"):
        lines = response_text.splitlines()
        if len(lines) >= 3:
            response_text = "\n".join(lines[1:-1]).strip()

    # -------------------------------
    # 3.7) Intentar parsear JSON
    # -------------------------------
    try:
        resultados_json = json.loads(response_text)
    except json.JSONDecodeError:
        resultados_json = {
            "error": "No se pudo parsear JSON",
            "raw_response": response_text
        }

    # -------------------------------
    # 3.8) Limpiar archivo temporal
    # -------------------------------
    os.remove(tmp_path)

    # -------------------------------
    # 3.9) Devolver la respuesta
    # -------------------------------
    return {
        "municipio": municipio,
        "pagina_inicio": pagina_inicio,
        "pagina_fin": pagina_fin,
        "resultados": resultados_json,
    }


# ------------------------------------------------------------
# 4) Si ejecutaras FastAPI directamente (uvicorn), dejarías algo así:
#    uvicorn function_app:app --host 0.0.0.0 --port 8000 --workers 1
# ------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("function_app:app", host="0.0.0.0", port=8002, workers=1)
