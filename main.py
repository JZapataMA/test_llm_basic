import os
from pdf2image import convert_from_path
import pytesseract
from langchain.docstore.document import Document
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from utils import buscar_cita_en_paginas
import warnings



load_dotenv()

import warnings
warnings.filterwarnings("ignore")

# Definición de paths para guardar las respuestas
respuestas_paths = {
    "a": "resumen.txt",
    "b": "tono.txt",
    "c": "cita_mario.txt",
    "d": "cita_miranda.txt",
    "e": "firmantes.txt"
}
os.makedirs("respuestas", exist_ok=True)
respuestas_paths = {k: os.path.join("respuestas", v) for k, v in respuestas_paths.items()}
print(respuestas_paths)

# Configuración del token de Hugging Face
api_token = os.getenv("API-KEY")

# Carga y preparación del documento
DOC_PATH = "Documento.pdf"

if not os.path.exists("pages.txt"):

    pages_images = convert_from_path(DOC_PATH, dpi=300)

    # Para cada imagen, aplica OCR y crea un objeto Document (de LangChain)
    documents = []
    for i, image in enumerate(pages_images):
        # Ajusta el parámetro 'lang' según el idioma del documento (por ejemplo, 'spa' para español)
        text = pytesseract.image_to_string(image, lang='spa')
        documents.append(Document(page_content=text, metadata={"page": i+1}))

    # Para búsquedas puntuales en páginas (como citas)
    pages = documents

# guardemos las pages en un archivo para leerlas más adelante

    with open("pages.txt", "w") as f:
        f.write(str(pages))

    print(f"Documento cargado con {len(pages)} páginas.")

else:
    with open("pages.txt", "r") as f:
        pages = eval(f.read())
    print(f"Documento cargado con {len(pages)} páginas.")




# Configuración de las preguntas
preguntas = {
    "resumen": "¿De qué trata el documento? Entregue un resumen de menos de 200 palabras.",
    "tono": "¿Cuál es el tono del documento, positivo o negativo?",
    "mario": "¿En qué página del documento se cita a Mario Pérez?",
    "miranda": "¿En qué página del documento se cita a doña Miranda?",
    "firmantes": "¿Hay firmantes del documento? En caso que sí, ¿Quiénes son?"
}

# Configuración del LLM con el modelo DeepSeek-R1-Distill-Qwen-32B vía Hugging Face
llm = HuggingFaceHub(
    repo_id="ibm-granite/granite-3.1-8b-instruct",
    huggingfacehub_api_token=api_token,
    model_kwargs={"temperature": 0.7}
)

# Definición del prompt base
prompt_template = """
A continuación se muestra un fragmento del documento:
--------------------------------------
{document_text}
--------------------------------------
Responda la siguiente pregunta:
{question}
"""
prompt = PromptTemplate(
    input_variables=["document_text", "question"],
    template=prompt_template
)

chain = LLMChain(llm=llm, prompt=prompt)

# --- Manejo de documentos extensos: dividir el contenido ---
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Unir todo el contenido del documento
documento_completo = "\n".join([doc.page_content for doc in pages])

# Dividir el documento en chunks manejables
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=200
)
chunks = text_splitter.split_text(documento_completo)
print(f"Documento dividido en {len(chunks)} fragmentos.")

# Ejemplo: generar un resumen parcial para cada chunk
resumen_parciales = []
for chunk in chunks:
    resumen_chunk = chain.run(document_text=chunk, question=preguntas["resumen"])
    resumen_parciales.append(resumen_chunk)

# Luego, se pueden combinar los resúmenes parciales para generar un resumen global
documento_resumido = "\n".join(resumen_parciales)
resumen_global = chain.run(document_text=documento_resumido, question="Resume lo siguiente en menos de 200 palabras:")
print("Resumen global del documento:")
print(resumen_global)

with open(respuestas_paths["a"], "w") as f:
    f.write(resumen_global)

# Tono del documento
tono = chain.run(document_text=documento_completo, question=preguntas["tono"])
print("Tono del documento:")
print(tono)
with open(respuestas_paths["b"], "w") as f:
    f.write(tono)

# Búsqueda de citas (puedes continuar usando la función ya definida)
paginas_mario = buscar_cita_en_paginas(pages, "Mario Pérez")
with open(respuestas_paths["c"], "w") as f:
    f.write(str(paginas_mario))

paginas_miranda = buscar_cita_en_paginas(pages, "doña Miranda")
with open(respuestas_paths["d"], "w") as f:
    f.write(str(paginas_miranda))

# Firmantes del documento
firmantes = chain.run(document_text=documento_completo, question=preguntas["firmantes"])
with open(respuestas_paths["e"], "w") as f:
    f.write(firmantes)

print("Proceso de extracción de información finalizado.")
print("Respuestas guardadas en la carpeta 'respuestas'.")
