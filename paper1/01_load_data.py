#%% Librerías

# Librerías úitiles
import os
import pandas as pd
import datetime as dt

# Librerías para cargar datos
from utils.load_data_energia import load_data as get_data_energia
from utils.load_data_futuros import get_data_futuros

# Librerías para procesar datos
from utils.preprocesamiento_datos import (
    procesar_precios,
    procesar_precios_ponderados,
    procesar_bilaterales
)

#%% Configuración
class Config:
    RAW_DATA_PATH = os.path.join(os.getcwd(), "data/1_raw")
    SILVER_DATA_PATH = os.path.join(os.getcwd(), "data/2_curated")

    DICT_XM_FILES = {
        "PRECIOS": "datos_crudos_PRECIOS.csv",
        "PRECIOS_PONDERADOS": "datos_crudos_PRECIOS_PONDERADOS.csv",
        "PRECIOS_BILATERALES": "datos_crudos_PRECIOS_BILATERALES.csv"
    }

#%% Funciones

def load_data_files(
        fecha_inicio_str: str,
        fecha_fin_str: str
    ) -> None:
    print("Iniciando carga de datos...")
    
    # Cargar datos de XM
    get_data_energia(
        fecha_inicio_str=fecha_inicio_str,
        fecha_fin_str=fecha_fin_str,
        data_path=Config.RAW_DATA_PATH
    )
    
    # Procesar datos de XM
    print("Procesando datos de XM...")
    for nombre, file_name in Config.DICT_XM_FILES.items():
        df = pd.read_csv(os.path.join(Config.RAW_DATA_PATH, file_name))
        
        if nombre == "PRECIOS":
            print("  Procesando precios...")
            df = procesar_precios(df.copy())
            df.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_{nombre}.csv"), index=False)
        
        if nombre == "PRECIOS_PONDERADOS":
            print("  Procesando precios ponderados...")
            df = procesar_precios_ponderados(df.copy())
            df.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_{nombre}.csv"), index=False)
        
        if nombre == "PRECIOS_BILATERALES":
            print("  Procesando precios bilaterales...")
            df = procesar_bilaterales(df.copy())
            df.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_{nombre}.csv"), index=False)

    print("Datos de XM procesados y guardados.")

    # Cargar datos de futuros
    df_futuros, df_futuros_wide = get_data_futuros()
    df_futuros = df_futuros[df_futuros["Fecha"] <= fecha_fin_str]
    df_futuros_wide = df_futuros_wide[df_futuros_wide["Fecha"] <= fecha_fin_str]
    df_futuros.to_csv(os.path.join(Config.SILVER_DATA_PATH, "precios_FUTUROS.csv"), index=False)
    df_futuros_wide.to_csv(os.path.join(Config.SILVER_DATA_PATH, "precios_FUTUROS_WIDE.csv"), index=False)

    print("Proceso completado.")
    return None


#%% Ejecutar carga de datos
if __name__ == "__main__":
    fecha_inicio_str = "2022-01-01"
    #fecha_fin_str = (dt.datetime.now() - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    fecha_fin_str = "2026-03-01"
    load_data_files(fecha_inicio_str, fecha_fin_str)