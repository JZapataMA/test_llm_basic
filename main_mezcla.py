"""
==============================================================
PROCESAMIENTO Y CONSULTA DE UN DOCUMENTO PDF CON OCR Y QA
==============================================================

Este script realiza lo siguiente:
1. Extrae el texto de un PDF mediante OCR (pytesseract) y lo guarda para evitar reprocesos.
2. Une el contenido de todas las páginas en un único string.
3. Divide el contenido en fragmentos y genera embeddings utilizando un modelo multilingüe de Hugging Face.
4. Configura un sistema QA que, mediante un LLM de Watsonx, responde a consultas definidas (en español).
5. Guarda las respuestas obtenidas en archivos.
"""

#########################
# 1. CONFIGURACIÓN Y CARGA DE LIBRERÍAS
#########################
import os
import warnings
from dotenv import load_dotenv

# Cargar variables de entorno y configurar warnings
load_dotenv()
warnings.filterwarnings("ignore")

# Importaciones para procesamiento de PDF y OCR
from pdf2image import convert_from_path
import pytesseract

# Importaciones de LangChain y utilidades
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Importaciones para el LLM de Watsonx y utilidades IBM
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_granite_community.notebook_utils import get_env_var

# Importación de vectorstore alternativo (Chroma) y función de búsqueda de citas
from langchain_community.vectorstores import Chroma
from utils import buscar_cita_en_paginas, clean_document, download_nltk_resources


download_nltk_resources()
#########################
# 2. CONFIGURACIÓN DE RUTAS Y CREDENCIALES
#########################
# Definir rutas para guardar respuestas
respuestas_paths = {
    "a": "resumen.txt",
    "b": "tono.txt",
    "c": "cita_mario.txt",
    "d": "cita_miranda.txt",
    "e": "firmantes.txt",
    "f": "cita_mario_llm.txt",
    "g": "cita_miranda_llm.txt"
}
os.makedirs("respuestas", exist_ok=True)
respuestas_paths = {k: os.path.join("respuestas", v) for k, v in respuestas_paths.items()}
print("Rutas de respuesta:", respuestas_paths)

# Configuración de credenciales
api_token = os.getenv("API-KEY")
watsonx_url = os.getenv("WATSONX_URL")
watsonx_apikey = os.getenv("WATSONX_APIKEY")
watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")
project_id = get_env_var("WATSONX_PROJECT_ID")

# Definir el path del documento PDF
DOC_PATH = "Documento.pdf"

#########################
# 3. EXTRAER TEXTO DEL PDF CON OCR
#########################

if not os.path.exists("pages.txt"):
    pages_images = convert_from_path(DOC_PATH, dpi=100)
    documents = []
    for i, image in enumerate(pages_images):
        # 'lang' se establece en 'spa' para procesar en español
        text = pytesseract.image_to_string(image, lang='spa')
        documents.append(Document(page_content=text, metadata={"page": i+1}))
    with open("pages.txt", "w", encoding="utf-8") as f:
        f.write(str(documents))
    print(f"Documento procesado y guardado con {len(documents)} páginas.")
else:
    with open("pages.txt", "r", encoding="utf-8") as f:
        documents = eval(f.read())
        # procesar el texto para eliminar caracteres especiales y stopwords
        documents = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in documents]

    print(f"Documento cargado con {len(documents)} páginas desde pages.txt.")

# Unir todo el contenido en un único string
#documento_completo = "\n".join([doc.page_content for doc in documents])

documento_completo = "\n".join([
    f"Página {doc.metadata.get('page', 'Desconocido')}: {doc.page_content}"
    for doc in documents
])


#########################
# 4. CONFIGURACIÓN DE CONSULTAS (PREGUNTAS)
#########################
preguntas = {
    "resumen": "¿De qué trata el documento? Entregue un resumen de menos de 200 palabras.",
    "tono": "¿Cuál es el tono del documento, positivo o negativo?",
    "mario": "¿En qué página del documento se cita a Mario Pérez?",
    "miranda": "¿En qué página del documento se cita a doña MIRANDA?",
    "firmantes": "¿Hay firmantes del documento? En caso que sí, ¿Quiénes son?"
}

#########################
# 5. DIVIDIR EL DOCUMENTO Y CREAR EMBEDDINGS
#########################
# Dividir el documento en fragmentos manejables
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
chunks = text_splitter.split_text(documento_completo)
print(f"Documento dividido en {len(chunks)} fragmentos.")

# Crear objetos Document para cada fragmento
chunk_docs = [
    Document(
        page_content=chunk,
        metadata={"page": doc.metadata.get("page")}
    )
    for doc in documents
    for chunk in text_splitter.split_text(doc.page_content)
]

# Crear embeddings utilizando HuggingFaceEmbeddings (modelo multilingüe que soporta español)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Construir vectorstore FAISS a partir de los documentos fragmentados
vectorstore = FAISS.from_documents(chunk_docs, embeddings)

#########################
# 6. CONFIGURACIÓN DEL MODELO DE LENGUAJE Y PROMPT
#########################

llm = WatsonxLLM(
    model_id="ibm/granite-3-2-8b-instruct-preview-rc",
    url=watsonx_url,
    apikey=watsonx_apikey,
    project_id=project_id,
    params={
        GenParams.DECODING_METHOD: "greedy",
        GenParams.TEMPERATURE: 0.7,
        GenParams.MIN_NEW_TOKENS: 5,
        GenParams.MAX_NEW_TOKENS: 8192,
        
    },
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Utiliza el siguiente contexto para responder a la pregunta en español. "
        "Responde únicamente con la respuesta, sin comentarios adicionales.\n\n"
        "Contexto: {context}\n"
        "Pregunta: {question}\n"
        "Respuesta:"
    )
)

# Crear el sistema QA utilizando RetrievalQA con el vectorstore FAISS
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt_template}
)

#########################
# 7. DEFINICIÓN DE FUNCIONES PARA CONSULTAS
#########################
def obtener_resumen():
    """Obtiene un resumen del documento en menos de 200 palabras."""
    return qa.invoke(preguntas["resumen"])["result"]

def obtener_tono():
    """Obtiene el tono del documento (positivo o negativo)."""
    return qa.invoke(preguntas["tono"])["result"]

def obtener_cita_mario():
    """Obtiene la(s) página(s) donde se cita a 'Mario Pérez'."""
    return buscar_cita_en_paginas(documents, "Mario Pérez")

def obtener_cita_miranda():
    """Obtiene la(s) página(s) donde se cita a 'doña Miranda'."""
    return buscar_cita_en_paginas(documents, "doña Miranda")

def obtener_firmantes():
    """Obtiene los firmantes del documento."""
    return qa.invoke(preguntas["firmantes"])["result"]

def obtener_cita_mario_llm():
    resultado = qa.invoke({"query": preguntas["mario"]})
    return resultado["result"]

def obtener_cita_miranda_llm():
    resultado = qa.invoke({"query": preguntas["miranda"]})
    return resultado["result"]

#########################
# 8. FUNCIÓN PRINCIPAL (MAIN)
#########################
def main():
    resumen = obtener_resumen()
    tono = obtener_tono()
    cita_mario = obtener_cita_mario()
    cita_miranda = obtener_cita_miranda()
    firmantes = obtener_firmantes()
    cita_mario_llm = obtener_cita_mario_llm()
    cita_miranda_llm = obtener_cita_miranda_llm()


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
    with open(respuestas_paths["f"], "w", encoding="utf-8") as f:
        f.write(cita_mario_llm)
    with open(respuestas_paths["g"], "w", encoding="utf-8") as f:
        f.write(cita_miranda_llm)

    print("Respuestas guardadas en la carpeta 'respuestas'.")

if __name__ == "__main__":
    main()
