import requests
from bs4 import BeautifulSoup

URL_TRANSPARENCIA = "https://www.ucss.edu.pe/nosotros/transparencia"

def extraer_pagina_principal():
    print(f"Descargando contenido de: {URL_TRANSPARENCIA}")
    # Desactivamos verificación SSL si hay problemas, o la dejamos por defecto
    respuesta = requests.get(URL_TRANSPARENCIA, verify=False)
    
    if respuesta.status_code != 200:
        print("Error al acceder a la página.")
        return
        
    soup = BeautifulSoup(respuesta.text, "html.parser")
    
    # Extraer todos los enlaces
    enlaces = soup.find_all("a")
    print(f"\nSe encontraron {len(enlaces)} enlaces en la página principal.")
    
    documentos = []
    for enlace in enlaces:
        texto = enlace.get_text(strip=True)
        href = enlace.get("href")
        
        # Filtramos un poco para buscar enlaces que parezcan documentos o subsecciones
        if texto and href and not href.startswith("#"):
            documentos.append({"titulo": texto, "enlace": href})
            
    print("\nEjemplos de información/documentos encontrados:")
    for doc in documentos[:15]:
        print(f"- {doc['titulo']} -> {doc['enlace']}")

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings()
    extraer_pagina_principal()
