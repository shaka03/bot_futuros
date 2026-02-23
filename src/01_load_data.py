#%% Librerías

# Librerías úitiles
import os
import pandas as pd
import datetime as dt
import unicodedata
import re

# Librerías para cargar datos
from utils.load_data_energia import load_data as get_data_energia
from utils.load_data_futuros import get_data_futuros

# Librerías para procesar datos
from utils.preprocesamiento_datos import (
    procesar_demanda,
    procesar_precios,
    procesar_precios_ponderados,
    procesar_bilaterales,
    procesar_aportes_hidricos,
    procesar_niveles_embalse,
    procesar_disponibilidad,
    procesar_generacion
)

from utils.get_news import main_process
from utils.model_news import process_news

#%% Configuración
class Config:
    RAW_DATA_PATH = os.path.join(os.getcwd(), "data/1_raw")
    SILVER_DATA_PATH = os.path.join(os.getcwd(), "data/2_silver")

    DICT_XM_FILES = {
        "DEMANDA": "datos_crudos_DEMANDA.csv",
        "PRECIOS": "datos_crudos_PRECIOS.csv",
        "PRECIOS_PONDERADOS": "datos_crudos_PRECIOS_PONDERADOS.csv",
        "APORTES_HIDRICOS": "datos_crudos_APORTES_HIDRICOS.csv",
        #"NIVELES_EMBALSE": "datos_crudos_NIVELES_EMBALSE.csv",
        "DISPONIBILIDAD_REAL": "datos_crudos_DISPONIBILIDAD_REAL.csv",
        "GENERACION_REAL": "datos_crudos_GENERACION_REAL.csv",
        "PRECIOS_BILATERALES": "datos_crudos_PRECIOS_BILATERALES.csv"
    }

#%% Funciones

def limpiar_fuente_en_titulo(texto):
    if pd.isna(texto): return texto
    # Dividimos por el último guion encontrado
    partes = str(texto).rsplit(' - ', 1)
    return partes[0].strip()

def normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_news(raw_data_path: str) -> None:
    print("Procesando noticias...")
    
    # 1. Cargar datos de noticias
    df_noticias = pd.read_csv(os.path.join(raw_data_path, "datos_crudos_NOTICIAS.csv"))

    # 2. Conversión de fecha (formato Google News)
    df_noticias['Fecha'] = pd.to_datetime(df_noticias['Fecha'], utc=True, errors='coerce')

    # 3. Crear columna Hora (antes de eliminarla de Fecha)
    df_noticias['Hora'] = df_noticias['Fecha'].dt.strftime('%H:%M:%S')

    # 4. Dejar Fecha únicamente como fecha (sin hora)
    df_noticias['Fecha'] = df_noticias['Fecha'].dt.date
    df_noticias['Fecha'] = pd.to_datetime(df_noticias['Fecha'])

    # 5. Diccionarios para español
    dias = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
        'Thursday': 'Jueves', 'Friday': 'Viernes',
        'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }

    meses = {
        'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo',
        'April': 'Abril', 'May': 'Mayo', 'June': 'Junio',
        'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre',
        'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'
    }

    # 6. Variables temporales
    df_noticias['Año'] = df_noticias['Fecha'].dt.year
    df_noticias['Mes'] = df_noticias['Fecha'].dt.month_name().map(meses)
    df_noticias['Dia'] = df_noticias['Fecha'].dt.day_name().map(dias)

    # 7. Eliminación de duplicados
    df_noticias = df_noticias.drop_duplicates(
        subset=['Titulo', 'Fuente', 'Fecha']
    ).reset_index(drop=True)

    # 8. Limpieza mínima
    df_noticias = df_noticias.dropna(subset=['Titulo', 'Fecha'])

    # 9. Normalización de títulos
    df_noticias["Titulo_norm"] = df_noticias["Titulo"].astype(str).apply(limpiar_fuente_en_titulo)
    df_noticias["Titulo_norm"] = df_noticias["Titulo_norm"].astype(str).apply(normalize_text)

    # Etiquetado de datos
    df_noticias = process_news(df_noticias)

    # Agregar a los datos históricos
    data_hist_path = os.path.join(Config.SILVER_DATA_PATH, "datos_NOTICIAS.csv")
    if os.path.exists(data_hist_path):
        df_noticias_hist = pd.read_csv(data_hist_path)
        df_noticias_hist['Fecha'] = pd.to_datetime(df_noticias_hist['Fecha'], errors='coerce')
        df_final = pd.concat([df_noticias_hist, df_noticias], ignore_index=True)
    else:
        df_final = df_noticias.copy()

    cols_keep = [
        "Titulo", "Fuente", "Fecha", "Hora", "url", "Titulo_norm", "Tipo_noticia"
    ]
    df_final = df_final[cols_keep].drop_duplicates(subset=["url"]).reset_index(drop=True)

    return df_final

def load_data_files(
        fecha_inicio_str: str,
        fecha_fin_str: str
    ) -> None:
    print("Iniciando carga de datos...")
    
    # Cargar datos de XM
    #get_data_energia(
    #    fecha_inicio_str=fecha_inicio_str,
    #    fecha_fin_str=fecha_fin_str,
    #    data_path=Config.RAW_DATA_PATH
    #)
    
    # Procesar datos de XM
    print("Procesando datos de XM...")
    for nombre, file_name in Config.DICT_XM_FILES.items():
        df = pd.read_csv(os.path.join(Config.RAW_DATA_PATH, file_name))
        if nombre == "DEMANDA":
            print("  Procesando demanda...")
            df_demanda, df_comprador = procesar_demanda(df.copy())
            df_demanda.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_{nombre}.csv"), index=False)
            df_comprador.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_DEMANDA_COMPRADOR.csv"), index=False)
        
        if nombre == "PRECIOS":
            print("  Procesando precios...")
            df = procesar_precios(df.copy())
            df.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_{nombre}.csv"), index=False)
        
        if nombre == "PRECIOS_PONDERADOS":
            print("  Procesando precios ponderados...")
            df = procesar_precios_ponderados(df.copy())
            df.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_{nombre}.csv"), index=False)
        
        if nombre == "APORTES_HIDRICOS":
            print("  Procesando aportes hídricos...")
            df = procesar_aportes_hidricos(df.copy())
            df.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_{nombre}.csv"), index=False)
        
        if nombre == "NIVELES_EMBALSE":
            print("  Procesando niveles de embalse...")
            df = procesar_niveles_embalse(df.copy())
            df.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_{nombre}.csv"), index=False)

        if nombre == "DISPONIBILIDAD_REAL":
            print("  Procesando disponibilidad real...")
            df = procesar_disponibilidad(df.copy())
            df.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_{nombre}.csv"), index=False)

        if nombre == "GENERACION_REAL":
            print("  Procesando generación real...")
            df = procesar_generacion(df.copy())
            df.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_{nombre}.csv"), index=False)
            
        if nombre == "PRECIOS_BILATERALES":
            print("  Procesando precios bilaterales...")
            df = procesar_bilaterales(df.copy())
            df.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_{nombre}.csv"), index=False)

    print("Datos de XM procesados y guardados.")

    # Cargar datos de futuros
    df_futuros = get_data_futuros()
    df_futuros = df_futuros[df_futuros["Fecha"] <= fecha_fin_str]
    df_futuros.to_csv(os.path.join(Config.SILVER_DATA_PATH, "precios_FUTUROS.csv"), index=False)

    # Cargar noticias
    #print("Cargando noticias...")
    #main_process(Config.RAW_DATA_PATH)
    #df_noticias = process_news(Config.RAW_DATA_PATH)
    #df_noticias = df_noticias[df_noticias["Fecha"] <= fecha_fin_str]
    #df_noticias.to_csv(os.path.join(Config.SILVER_DATA_PATH, "datos_NOTICIAS.csv"), index=False)

    print("Proceso completado.")
    return None


#%% Ejecutar carga de datos
if __name__ == "__main__":
    fecha_inicio_str = "2022-01-01"
    #fecha_fin_str = (dt.datetime.now() - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    fecha_fin_str = "2026-01-31"
    load_data_files(fecha_inicio_str, fecha_fin_str)