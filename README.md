# Procesamiento de Documentos PDF y Sistema de QA con LLM

Esta propuesta de solución al desafío Arkho implementa un flujo de trabajo para procesar documentos PDF y construir un sistema de Preguntas & Respuestas (QA) utilizando técnicas de OCR, embeddings y un modelo LLM. La idea principal es extraer el contenido textual del PDF, segmentarlo, vectorizarlo y finalmente emplear un modelo de lenguaje para responder consultas específicas acerca del documento.

---

## Tabla de Contenidos

- [Descripción del Flujo de Trabajo](#descripción-del-flujo-de-trabajo)
- [Pasos del Proceso](#pasos-del-proceso)
- [Función de Utilidad: `buscar_cita_en_paginas`](#función-de-utilidad-buscar_cita_en_paginas)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Código](#estructura-del-código)
- [Inspiración](#inspiración)

---

## Descripción del Flujo de Trabajo

El proyecto se inspira en una presentación de [IBM Community](https://github.com/ibm-granite-community) que muestra el poder del modelo de lenguaje Granite de IBM. A partir de esa base, se ha desarrollado un flujo que abarca desde la extracción de texto mediante OCR hasta la generación de respuestas mediante un sistema QA que combina embeddings y un LLM.

---

## Pasos del Proceso

1. **Extracción del Texto mediante OCR (Tesseract):**  
   - Se convierte cada página del PDF en una imagen utilizando `pdf2image`.
   - Se aplica [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) para extraer el texto en español de cada imagen.
   - El resultado se almacena en un archivo temporal (`pages.txt`) para evitar reprocesos en futuras ejecuciones.

2. **Unificación del Contenido:**  
   - Se concatena el texto de todas las páginas en un único string que representa el documento completo.

3. **División del Documento en Fragmentos (Chunks):**  
   - Se utiliza `RecursiveCharacterTextSplitter` de LangChain para dividir el documento en partes más pequeñas, optimizando el procesamiento y evitando superar los límites de tokens del modelo.

4. **Vectorización (Embeddings) de Cada Fragmento:**  
   - Se emplea el modelo de embeddings [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) para transformar cada fragmento en un vector de alta dimensión.
   - Estos vectores permiten realizar búsquedas semánticas de forma eficiente.

5. **Creación de la Base de Datos Vectorial con FAISS:**  
   - Los embeddings generados se indexan en una base de datos vectorial FAISS para permitir búsquedas por similitud.

6. **Configuración del LLM y del Sistema QA:**  
   - Se configura un modelo LLM (en este caso, el modelo de Watsonx de IBM) para generar respuestas en español.
   - Se define un prompt template que obliga a responder únicamente con la respuesta, sin comentarios adicionales.
   - Se construye el sistema QA utilizando `RetrievalQA`, que combina la búsqueda de los fragmentos más relevantes en FAISS y la generación de respuestas con el LLM.

7. **Definición de Funciones Específicas de Consulta:**  
   - Se implementan funciones para responder preguntas concretas, tales como:
     - **Resumen del Documento**
     - **Tono del Documento**
     - **Firmantes del Documento**
     - **Páginas donde se cita a "Mario Pérez" y "doña Miranda"**  
     
   - La función `buscar_cita_en_paginas` se utiliza para identificar en qué páginas aparece una cita específica.

8. **Ejecución Principal (`main`):**  
   - Se invocan las funciones de consulta y se guardan las respuestas en archivos de texto dentro de la carpeta `respuestas`.

---

## Función de Utilidad: `buscar_cita_en_paginas`

Esta función recorre una lista de objetos `Document` (cada uno representa una página) y busca una palabra clave dentro del contenido, devolviendo el número de página donde se encontró la cita.

```python
def buscar_cita_en_paginas(documents, keyword, threshold=0.65):
    paginas = []
    for doc in documents:
        # Utiliza page_content si existe, de lo contrario, get_content()
        try:
            contenido = doc.page_content
        except AttributeError:
            contenido = doc.get_content() if hasattr(doc, "get_content") else ""
            
        # Dividir el contenido en palabras o frases para comparar
        palabras = contenido.lower().split()
        
        # Usar difflib para calcular la similaridad entre palabras
        for i in range(len(palabras)):
            # Comprobamos palabras individuales y combinaciones de dos palabras
            candidatos = [palabras[i]]
            if i < len(palabras) - 1:
                candidatos.append(f"{palabras[i]} {palabras[i+1]}")
            
            for candidato in candidatos:
                ratio = difflib.SequenceMatcher(None, candidato, keyword.lower()).ratio()
                if ratio >= threshold:
                    # Intenta obtener el número de página desde metadata, si existe
                    pagina = doc.metadata.get("page", "Desconocido")
                    if pagina not in paginas:  # Evitar duplicados
                        paginas.append(pagina)
                    break
    
    return paginas if paginas else "No se encontró la cita."
```


## Inspiración

El diseño y flujo de trabajo de este proyecto se basa en una presentación de IBM Community, en la cual se muestra el uso y el poder de Granite, el modelo de lenguaje de IBM. Este proyecto adapta esos conceptos para procesar documentos PDF y responder preguntas específicas utilizando tecnologías de vanguardia en procesamiento de lenguaje natural.

## Uso

Para ejecutar el flujo de trabajo y obtener las respuestas a las consultas definidas, simplemente ejecuta:

```
python main.py
```

Esto:

- Procesará el documento PDF mediante OCR.
- Extraerá, unificará y dividirá el contenido.
- Generará los embeddings y creará la base de datos vectorial.
- Ejecutará las consultas mediante el sistema QA y guardará las respuestas en la carpeta respuestas.


