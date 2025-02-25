import difflib
import re
import string
import unicodedata
from nltk.corpus import stopwords
import nltk
from spellchecker import SpellChecker

def buscar_cita_en_paginas(documents, keyword, threshold=0.8):
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

# Descargar recursos necesarios de NLTK (ejecutar una vez)
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

def preprocess_text(text, remove_stopwords=False):
    """
    Preprocesa el texto para NLP manteniendo símbolos relevantes como $ y €.
    
    Args:
        text (str): Texto a procesar
        remove_stopwords (bool): Si se deben eliminar stopwords
    
    Returns:
        str: Texto procesado
    """
    if not text:
        return ""
    
    # Normalizar unicode y convertir a minúsculas
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    
    # Preservar símbolos monetarios y otros importantes
    symbols_to_keep = ['$', '€', '%', '¢', '£', '¥', '₡', '₤', '₦', '₩', '₪', '₫', '₭', '₮', '₱', '₲', '₴', '₸']
    
    # Reemplazar símbolos a preservar con placeholders
    placeholder_map = {}
    for i, symbol in enumerate(symbols_to_keep):
        placeholder = f"SYMBOL{i}"
        placeholder_map[placeholder] = symbol
        text = text.replace(symbol, f" {placeholder} ")
    
    # Eliminar caracteres especiales innecesarios pero mantener puntuación esencial
    text = re.sub(r'[^\w\s.,;:!?¿¡()\[\]{}\-_"\']+', ' ', text)
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar stopwords si se solicita
    if remove_stopwords:
        stop_words = set(stopwords.words('spanish'))
        words = text.split()
        text = ' '.join([word for word in words if word not in stop_words])
    
    # Restaurar símbolos originales
    for placeholder, symbol in placeholder_map.items():
        text = text.replace(placeholder, symbol)
    
    return text.strip()

def filter_spanish_words(text):
    """
    Filtra el texto para mantener solo palabras válidas en español.
    
    Args:
        text (str): Texto a filtrar
    
    Returns:
        str: Texto con palabras filtradas
    """
    download_nltk_resources()
    
    # Inicializar corrector ortográfico en español
    spell = SpellChecker(language='es')
    
    words = text.split()
    filtered_words = []
    
    for word in words:
        # Saltarse símbolos y números
        if not word.isalpha():
            filtered_words.append(word)
            continue
            
        # Mantener palabras cortas (2-3 letras) sin verificar
        if len(word) <= 3:
            filtered_words.append(word)
            continue
            
        # Verificar si es una palabra válida en español
        if word in spell or word.lower() in spell:
            filtered_words.append(word)
    
    return ' '.join(filtered_words)

def clean_document(text):
    """
    Aplica toda la limpieza y filtrado al documento.
    """
    # Aplicar preprocesamiento básico
    text = preprocess_text(text)
    
    # Filtrar para mantener solo palabras en español
    text = filter_spanish_words(text)
    
    return text