import nest_asyncio
nest_asyncio.apply()

import os
from dotenv import load_dotenv
load_dotenv()

# --- PARSEO DEL DOCUMENTO CON LLAMAPARSE Y CACHE ---
PARSED_CACHE = "documento_parseado.txt"

if not os.path.exists(PARSED_CACHE):
    from llama_cloud_services import LlamaParse
    from llama_index.core import SimpleDirectoryReader

    # Configura el parser para retornar en formato markdown
    parser = LlamaParse(result_type="text", language="es")
    file_extractor = {".pdf": parser}

    # Procesa el PDF usando LlamaParse (esto crea una lista de documentos parseados)
    parsed_documents = SimpleDirectoryReader(input_files=['Documento.pdf'], file_extractor=file_extractor).load_data()
    print("Documento parseado con LlamaParse. Cantidad de documentos:", len(parsed_documents))

    # Combina el contenido parseado; se asume que cada documento tiene atributo 'text' o, si no, se usa get_content()
    documento_completo = "\n".join([getattr(doc, 'text', doc.get_content()) for doc in parsed_documents])
    
    # Guarda el contenido parseado para futuras ejecuciones
    with open(PARSED_CACHE, "w", encoding="utf-8") as f:
        f.write(documento_completo)
    print("Documento parseado guardado en", PARSED_CACHE)
    documents = parsed_documents
else:
    with open(PARSED_CACHE, "r", encoding="utf-8") as f:
        documento_completo = f.read()
    print("Documento parseado cargado desde", PARSED_CACHE)
    from langchain.docstore.document import Document
    documents = [Document(page_content=documento_completo, metadata={})]

# --- PROCESAMIENTO DEL DOCUMENTO PARA QA ---
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_granite_community.notebook_utils import get_env_var
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

# Definir paths para almacenar respuestas
respuestas_paths = {
    "a": "resumen.txt",
    "b": "tono.txt",
    "c": "cita_mario.txt",
    "d": "cita_miranda.txt",
    "e": "firmantes.txt"
}
os.makedirs("respuestas_optimizado", exist_ok=True)
respuestas_paths = {k: os.path.join("respuestas_optimizado", v) for k, v in respuestas_paths.items()}
print("Rutas de respuestas:", respuestas_paths)

# Credenciales de Watsonx
watsonx_url = os.getenv("WATSONX_URL")
watsonx_apikey = os.getenv("WATSONX_APIKEY")
watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")
project_id = get_env_var("WATSONX_PROJECT_ID")

# Configuración de las preguntas
preguntas = {
    "resumen": "¿De qué trata el documento? Entregue un resumen de menos de 200 palabras.",
    "tono": "¿Cuál es el tono del documento, positivo o negativo?",
    "mario": "¿En qué sección del documento se cita a Mario Pérez?",
    "miranda": "¿En qué sección del documento se cita a doña Miranda?",
    "firmantes": "¿Hay firmantes del documento? En caso que sí, ¿Quiénes son?"
}

# Dividir el contenido parseado en fragmentos manejables
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=256, chunk_overlap=50
)
chunks = text_splitter.split_text(documento_completo)
print(f"Documento dividido en {len(chunks)} fragmentos.")

# Crear objetos Document para cada fragmento
chunk_docs = [Document(page_content=chunk) for chunk in chunks]

# Crear embeddings usando WatsonxEmbeddings (modelo IBM_SLATE_30M_ENG)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Construir la base de datos vectorial con Chroma
vectorstore = Chroma.from_documents(
    documents=chunk_docs,
    collection_name="agentic-rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

# Crear un prompt template para respuestas en español
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Utiliza únicamente el siguiente contexto extraído de un documento para responder la pregunta de manera concisa y precisa en español. "
        "Contexto: {context}\n"
        "Pregunta: {question}\n"
        "Respuesta:"
    )
)

# Configurar el LLM Watsonx
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
        GenParams.STOP_SEQUENCES: ["Human:", "Observation"],
    },
)

# Crear el sistema QA utilizando RetrievalQA con el prompt template definido
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
)

# --- Funciones para realizar consultas (de forma síncrona) ---
from utils import buscar_cita_en_paginas  # Asegúrate de tener implementada esta función

def obtener_resumen():
    resultado = qa.invoke({"query": preguntas["resumen"]})
    print(resultado)
    return resultado["result"]

def obtener_tono():
    resultado = qa.invoke({"query": preguntas["tono"]})
    print(resultado)
    return resultado["result"]

def obtener_cita_mario():
    return buscar_cita_en_paginas(documents, "Mario Pérez")

def obtener_cita_miranda():
    return buscar_cita_en_paginas(documents, "doña Miranda")

def obtener_firmantes():
    resultado = qa.invoke({"query": preguntas["firmantes"]})
    print(resultado)
    return resultado["result"]

def main():
    firmantes = obtener_firmantes()
    resumen = obtener_resumen()
    tono = obtener_tono()
    cita_mario = obtener_cita_mario()
    cita_miranda = obtener_cita_miranda()

    # Guardar las respuestas en archivos
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

    print("Respuestas guardadas en la carpeta 'respuestas_optimizado'.")

if __name__ == "__main__":
    main()
