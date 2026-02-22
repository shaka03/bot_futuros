#%% Librerías
import os
import pandas as pd
from gnews import GNews
import json
from datetime import datetime
import time

#%% Congiguración
class Config:
    queries = [
        "energía Colombia",
        "niveles embalses SIN Colombia XM",
        "aportaciones hídricas sistema interconectado nacional Colombia",
        "índice ENFICC generación hidroeléctrica Colombia",
        "alerta racionamiento energía Colombia fenómeno del Niño",
        "déficit hídrico generación eléctrica Colombia",
        "probabilidad racionamiento eléctrico Colombia 2026",
        "reservas hídricas XM reporte semanal",
        "precio bolsa energía XM histórico Colombia",
        "precio spot energía Colombia variación diaria",
        "precio escasez CREG Colombia actualización",
        "componente restricciones precio energía Colombia",
        "cargo por confiabilidad impacto tarifa energía Colombia",
        "volatilidad precio energía mercado mayorista Colombia",
        "incremento precio energía usuarios no regulados Colombia",
        "resolución CREG mercado mayorista energía",
        "modificación cargo por confiabilidad CREG",
        "reforma mercado eléctrico colombiano",
        "subastas cargo por confiabilidad Colombia",
        "regulación contratos bilaterales energía Colombia",
        "medidas gobierno crisis energética Colombia",
        "intervención tarifaria sector eléctrico Colombia",
        "crecimiento demanda energía Colombia UPME",
        "proyección demanda eléctrica Colombia 2026",
        "consumo energía sector industrial Colombia",
        "demanda máxima SIN Colombia récord",
        "racionamiento industrial energía Colombia",
        "contratos bilaterales energía Colombia precios",
        "precio contratos largo plazo energía Colombia",
        "mercado derivados energía Colombia",
        "coberturas precio energía Colombia industriales",
        "estrategias compra energía usuarios no regulados",
        "riesgo precio energía contratos Colombia",
        "hidrología embalses generación hidroeléctrica Colombia",
        "precio de escasez confiabilidad sistema eléctrico Colombia"
    ]


#%% Funciones
def load_last_update() -> datetime:
    try:
        with open("src/utils/update_news.json", "r") as f:
            data = json.load(f)
            last_update_str = data.get("last_update")
            if last_update_str:
                return datetime.strptime(last_update_str, "%Y-%m-%d")
    except FileNotFoundError:
        pass
    return None

def get_connection() -> GNews:
    google_news = GNews(
        language='es',
        country='CO',
        max_results=100        # Ajustable
    )
    return google_news

def generar_ventanas_mensuales(
        fecha_inicio: datetime,
        fecha_fin: datetime
) -> list[tuple[datetime, datetime]]:
    ventanas = []
    actual = fecha_inicio

    while actual < fecha_fin:
        if actual.month == 12:
            siguiente = datetime(actual.year + 1, 1, 1)
        else:
            siguiente = datetime(actual.year, actual.month + 1, 1)

        ventanas.append((actual, siguiente))
        actual = siguiente

    return ventanas

def obtener_noticias(
        google_news: GNews,
        queries_search: list[str],
        fecha_inicio: datetime,
        fecha_fin: datetime
) -> list[dict]:
    ventanas = generar_ventanas_mensuales(fecha_inicio, fecha_fin)
    data_new = []
    for query in queries_search:
        print(f"\n🔎 Query: {query}")

        for inicio, fin in ventanas:
            print(f"  ⏳ Ventana: {inicio} → {fin}")

            # Configurar ventana temporal
            google_news.start_date = inicio
            google_news.end_date = fin

            try:
                noticias = google_news.get_news(query)
            except Exception as e:
                print(f"   ⚠️ Error en ventana {inicio} → {fin}: {e}")
                continue

            # Procesar noticias
            for noticia in noticias:
                registro = {
                    "Query": query,
                    "Titulo": noticia.get("title"),
                    "Fuente": noticia.get("publisher", {}).get("title"),
                    "Fecha": noticia.get("published date"),
                    "url": noticia.get("url")
                }
                data_new.append(registro)

            # Pausa para evitar bloqueos
            time.sleep(1)

    return data_new

def save_to_csv(data: list[dict], filename: str):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\n✅ Datos guardados en {filename}")


def main_process(raw_data_path: str) -> None:
    google_news = get_connection()

    # Cargar última fecha de actualización
    last_update = load_last_update()
    if last_update:
        fecha_inicio = last_update
        fecha_fin = datetime.now()
    else:
        fecha_inicio = datetime(2015, 1, 1)
        fecha_fin = datetime.now()
    
    noticias = obtener_noticias(google_news, Config.queries, fecha_inicio, fecha_fin)
    save_to_csv(noticias, os.path.join(raw_data_path, "datos_crudos_NOTICIAS.csv"))