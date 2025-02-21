# Presentación del Flujo de Trabajo

A continuación se presenta un pequeño listado de puntos claves utilizados para poder procesar el documento y montar un QA a través del uso de un modelo LLM.
Para fuente de inspiración de la realización de este trabajo, utilice una presentación de [IBM Comunnity](https://github.com/ibm-granite-community) en donde enseñan el uso y el poder de granite, modelo de lenguaje de IBM.

## Pasos

1. **Extracción del texto mediante OCR (Tesseract)**
   - Se convierte cada página del PDF en imagen.  
   - Se aplica [Tesseract](https://github.com/tesseract-ocr/tesseract) para extraer el texto de cada imagen en español.  
   - Se almacena el resultado en un archivo temporal (`pages.txt`) para evitar reprocesar en cada ejecución.

2. **Unificación del contenido**
   - Se concatena el texto de todas las páginas en un solo _string_ (el documento completo).

3. **División del documento en fragmentos (Chunks)**
   - Se utiliza [`RecursiveCharacterTextSplitter`](https://api.python.langchain.com/en/latest/text_splitter.html#langchain.text_splitter.RecursiveCharacterTextSplitter) para fraccionar el texto extenso en partes más pequeñas y así optimizar el procesamiento, evitando superar límites de tokens en el modelo.

4. **Vectorización (Embeddings) de cada fragmento**
   - Se emplea el modelo de embeddings [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) para convertir cada _chunk_ en un vector de alta dimensión.
   - Estos vectores resultan esenciales para la búsqueda semántica.

5. **Creación de la base de datos vectorial con FAISS**
   - Se insertan los embeddings en una base [FAISS](https://github.com/facebookresearch/faiss), lo que permite realizar búsquedas por similitud de forma eficiente.

6. **Configuración de un LLM (Hugging Face Hub)**
   - Se utiliza el modelo [`ibm-granite/granite-3.1-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct) alojado en Hugging Face.  
   - Se define el token de acceso en el entorno y se configuran parámetros como la temperatura para el modelo.

7. **Construcción del sistema QA con RetrievalQA**
   - Se configura una instancia de [`RetrievalQA`](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.html) que:
     1. Recupera los fragmentos más relevantes usando FAISS (búsqueda por similitud).  
     2. Genera la respuesta final empleando el LLM.

8. **Definición de funciones específicas de consulta**
   - Se crean funciones para realizar preguntas concretas:
     - **Resumen**  
     - **Tono**  
     - **Firmantes**  
   - **Función `buscar_cita_en_paginas(paginas, cita)`**  

     ```python
     def buscar_cita_en_paginas(paginas, cita):
         paginas_encontradas = []
         for doc in paginas:
             # Se busca la cita ignorando mayúsculas/minúsculas
             if cita.lower() in doc.page_content.lower():
                 # Se obtiene el número de página de la metadata, si existe
                 numero_pagina = doc.metadata.get("page", "Desconocido")
                 paginas_encontradas.append(numero_pagina)
         return paginas_encontradas
     ```

     Esta función recorre cada objeto `Document` (correspondiente a una página) y verifica si el texto a buscar (`cita`) aparece en su contenido, ignorando mayúsculas y minúsculas. En caso de encontrar la coincidencia, recupera el número de página desde los metadatos del `Document`.  
     
     - **Páginas que mencionan a Mario Pérez**  
     - **Páginas que mencionan a doña Miranda**  

9. **Ejecución principal (`main`)**
   - Se llama a cada una de las funciones de consulta.  
   - Se guardan las respuestas en archivos de texto en la carpeta `respuestas` en un archivo `.txt`relativo a cada pregunta/respuesta.
