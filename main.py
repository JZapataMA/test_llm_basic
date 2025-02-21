import os
from pdf2image import convert_from_path
import pytesseract
from langchain.docstore.document import Document
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import warnings

from utils import buscar_cita_en_paginas

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

# Definir el path del documento PDF
DOC_PATH = "Documento.pdf"

# Extraer el texto del PDF mediante OCR y guardarlo en "pages.txt" para evitar reprocesos
if not os.path.exists("pages.txt"):
    pages_images = convert_from_path(DOC_PATH, dpi=300)
    documents = []
    for i, image in enumerate(pages_images):
        # Ajusta 'lang' según el idioma del documento, en este caso español ('spa')
        text = pytesseract.image_to_string(image, lang='spa')
        documents.append(Document(page_content=text, metadata={"page": i+1}))
    with open("pages.txt", "w", encoding="utf-8") as f:
        f.write(str(documents))
    print(f"Documento procesado y guardado con {len(documents)} páginas.")
else:
    with open("pages.txt", "r", encoding="utf-8") as f:
        documents = eval(f.read())
    print(f"Documento cargado con {len(documents)} páginas desde pages.txt.")

# Unir todo el contenido del documento en un solo string
documento_completo = "\n".join([doc.page_content for doc in documents])

# Configuración de las preguntas
preguntas = {
    "resumen": "¿De qué trata el documento? Entregue un resumen de menos de 200 palabras.",
    "tono": "¿Cuál es el tono del documento, positivo o negativo?",
    "mario": "¿En qué página del documento se cita a Mario Pérez?",
    "miranda": "¿En qué página del documento se cita a doña Miranda?",
    "firmantes": "¿Hay firmantes del documento? En caso que sí, ¿Quiénes son?"
}

# Dividir el documento en fragmentos manejables para generar embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,      # Tamaño máximo de cada fragmento
    chunk_overlap=50     # Solapamiento entre fragmentos para mantener contexto
)
chunks = text_splitter.split_text(documento_completo)
print(f"Documento dividido en {len(chunks)} fragmentos.")

# Crear objetos Document para cada fragmento
chunk_docs = [Document(page_content=chunk) for chunk in chunks]

# Crear embeddings utilizando el modelo all-MiniLM-L6-v2
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Crear la base de datos vectorial FAISS a partir de los documentos fragmentados
vectorstore = FAISS.from_documents(chunk_docs, embeddings)

# Configurar el modelo de lenguaje desde HuggingFaceHub
api_token = os.getenv("API-KEY")
llm = HuggingFaceHub(
    repo_id = "ibm-granite/granite-3.1-2b-instruct",
    huggingfacehub_api_token=api_token,
    model_kwargs={"temperature": 0.8}
)

# Creación del sistema QA utilizando RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Se puede experimentar con otros chain types, como "map_reduce"
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
)

# --- Definición de funciones para responder cada pregunta ---
def obtener_resumen():
    """Retorna un resumen del documento en menos de 200 palabras."""
    return qa.run(preguntas["resumen"])

def obtener_tono():
    """Retorna el tono del documento (positivo o negativo)."""
    return qa.run(preguntas["tono"])

def obtener_cita_mario():
    """Retorna la(s) página(s) donde se cita a 'Mario Pérez'."""
    return buscar_cita_en_paginas(documents, "Mario Pérez")

def obtener_cita_miranda():
    """Retorna la(s) página(s) donde se cita a 'doña Miranda'."""
    return buscar_cita_en_paginas(documents, "doña Miranda")

def obtener_firmantes():
    """Retorna los firmantes del documento, en caso de existir."""
    return qa.run(preguntas["firmantes"])

# --- Función principal para ejecutar todas las consultas ---
def main():
    resumen = obtener_resumen()
    tono = obtener_tono()
    cita_mario = obtener_cita_mario()
    cita_miranda = obtener_cita_miranda()
    firmantes = obtener_firmantes()

    # Guardar respuestas en archivos
    with open(respuestas_paths["a"], "w", encoding="utf-8") as f:
        f.write(resumen)
    with open(respuestas_paths["b"], "w", encoding="utf-8") as f:
        f.write(tono)
    with open(respuestas_paths["c"], "w", encoding="utf-8") as f:
        f.write(str(cita_mario))
    with open(respuestas_paths["d"], "w", encoding="utf-8") as f:
        f.write(str(cita_miranda))
    with open(respuestas_paths["e"], "w", encoding="utf-8") as f:
        f.write(firmantes)

    print("Respuestas guardadas en la carpeta 'respuestas'.")

if __name__ == "__main__":
    main()