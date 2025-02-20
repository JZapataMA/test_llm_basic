# -----------------------------------------------------------------------------
# Función para buscar en qué páginas se encuentra una cita dada
# -----------------------------------------------------------------------------
def buscar_cita_en_paginas(paginas, cita):
    paginas_encontradas = []
    for doc in paginas:
        # Se busca la cita ignorando mayúsculas/minúsculas
        if cita.lower() in doc.page_content.lower():
            # Se obtiene el número de página de la metadata, si existe.
            numero_pagina = doc.metadata.get("page", "Desconocido")
            paginas_encontradas.append(numero_pagina)
    return paginas_encontradas
