import os
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

from utils import buscar_cita_en_paginas

load_dotenv()

# -----------------------------------------------------------------------------
# Configuración de la API Key de OpenAI
# -----------------------------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------------------------------------------------------
# Carga y preparación del documento
# -----------------------------------------------------------------------------
DOC_PATH = "documento.pdf"

pdf_path = DOC_PATH
loader = PyPDFLoader(pdf_path)
# Cargamos el documento completo (la lista de Document, cada uno con metadata, e.g., número de página)
documents = loader.load()


pages = documents

# -----------------------------------------------------------------------------
# Definición de las preguntas a responder
# -----------------------------------------------------------------------------
preguntas = {
    "resumen": "¿De qué trata el documento? Entregue un resumen de menos de 200 palabras.",
    "tono": "¿Cuál es el tono del documento, positivo o negativo?",
    "mario": "¿En qué página del documento se cita a Mario Pérez?",
    "miranda": "¿En qué página del documento se cita a doña Miranda?",
    "firmantes": "¿Hay firmantes del documento? En caso que sí, ¿Quiénes son?"
}

# -----------------------------------------------------------------------------
# Configuración del LLM con langchain
# -----------------------------------------------------------------------------
llm = OpenAI(api_key=api_key, temperature=0.1)

# Se define un prompt template para procesar el documento completo junto con la pregunta.
prompt_template = """
A continuación se muestra el contenido completo del documento:
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

# Se combina el contenido completo del documento en un solo string.
documento_completo = "\n".join([doc.page_content for doc in documents])

# -----------------------------------------------------------------------------
# a. Resumen del documento (menos de 200 palabras)
# -----------------------------------------------------------------------------
resumen = chain.run(document_text=documento_completo, question=preguntas["resumen"])
print("Resumen del documento:")
print(resumen)
print("\n" + "="*60 + "\n")

# -----------------------------------------------------------------------------
# b. Tono del documento (positivo o negativo)
# -----------------------------------------------------------------------------
tono = chain.run(document_text=documento_completo, question=preguntas["tono"])
print("Tono del documento:")
print(tono)
print("\n" + "="*60 + "\n")

# -----------------------------------------------------------------------------
# c. Página(s) donde se cita a "Mario Pérez"
# -----------------------------------------------------------------------------
paginas_mario = buscar_cita_en_paginas(pages, "Mario Pérez")
print("Cita a 'Mario Pérez' encontrada en la(s) página(s):")
if paginas_mario:
    print(paginas_mario)
else:
    print("No se encontró cita a 'Mario Pérez'.")
print("\n" + "="*60 + "\n")

# -----------------------------------------------------------------------------
# d. Página(s) donde se cita a "doña Miranda"
# -----------------------------------------------------------------------------
paginas_miranda = buscar_cita_en_paginas(pages, "doña Miranda")
print("Cita a 'doña Miranda' encontrada en la(s) página(s):")
if paginas_miranda:
    print(paginas_miranda)
else:
    print("No se encontró cita a 'doña Miranda'.")
print("\n" + "="*60 + "\n")

# -----------------------------------------------------------------------------
# e. Firmantes del documento
# -----------------------------------------------------------------------------
firmantes = chain.run(document_text=documento_completo, question=preguntas["firmantes"])
print("Firmantes del documento:")
print(firmantes)
