def buscar_cita_en_paginas(documents, keyword):
    paginas = []
    for doc in documents:
        # Utiliza page_content si existe, de lo contrario, get_content()
        try:
            contenido = doc.page_content
        except AttributeError:
            contenido = doc.get_content() if hasattr(doc, "get_content") else ""
        if keyword.lower() in contenido.lower():
            # Intenta obtener el número de página desde metadata, si existe
            pagina = doc.metadata.get("page", "Desconocido")
            paginas.append(pagina)
    return paginas if paginas else "No se encontró la cita."
