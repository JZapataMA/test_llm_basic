import difflib
import re
import string
import unicodedata
from nltk.corpus import stopwords
import nltk
from spellchecker import SpellChecker

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