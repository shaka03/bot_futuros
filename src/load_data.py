#%% Librerías

# Librerías úitiles
import os
import pandas as pd
import datetime as dt

# Librerías para cargar datos
from load_data_energia import load_data as get_data_energia
from load_data_futuros import get_data_futuros

# Librerías para procesar datos
from limpieza_datos import (
    procesar_demanda,
    procesar_precios,
    procesar_precios_ponderados,
    procesar_aportes_hidricos,
    procesar_niveles_embalse,
    procesar_disponibilidad,
    procesar_generacion
)

#%% Configuración
class Config:
    RAW_DATA_PATH = os.path.join(os.getcwd(), "data/raw")
    SILVER_DATA_PATH = os.path.join(os.getcwd(), "data/silver")

#%% Funciones

def load_data_files():
    print("Iniciando carga de datos...")
    
    # Cargar datos de XM
    dict_xm_files =get_data_energia(
        fecha_inicio_str="2024-01-01",
        fecha_fin_str=dt.datetime.now().strftime("%Y-%m-%d"),
        data_path=RAW_DATA_PATH
    )

    # Procesar datos de XM
    print("Procesando datos de XM...")
    for nombre, file_name in dict_xm_files.items():
        df = pd.read_csv(os.path.join(Config.RAW_DATA_PATH, file_name))
        if nombre == "DEMANDA":
            print("  Procesando demanda...")
            df = procesar_demanda(df.copy())
            df.to_csv(os.path.join(Config.SILVER_DATA_PATH, f"datos_{nombre}.csv"), index=False)
        
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
            
    print("Datos de XM procesados y guardados.")

    # Cargar datos de futuros
    df_futuros = get_data_futuros()
    df_futuros.to_csv(os.path.join(Config.SILVER_DATA_PATH, "precios_FUTUROS.csv"), index=False)

    # Aquí puedes agregar más lógica para procesar o guardar los datos cargados
    print("Proceso completado.")
    return None


#%% Ejecutar carga de datos
if __name__ == "__main__":
    load_data_files()
