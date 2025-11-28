#%% Librerías

# Librerías úitiles
import os
import pandas as pd
import datetime as dt

# Librerías para cargar datos
from load_data_energia import load_data as get_data_energia
from load_data_futuros import get_data_futuros

#%% Configuración
DATA_PATH = os.path.join(os.getcwd(), "data/gold")

#%% Ejecución
def load_data_files():
    print("Iniciando carga de datos...")
    
    # Cargar datos de energía
    df_energia = get_data_energia(
        fecha_inicio_str="2024-01-01",
        fecha_fin_str=dt.datetime.now().strftime("%Y-%m-%d"),
    )
    df_energia.to_csv(f"{DATA_PATH}/datos_energia.csv", index=False)
    
    # Cargar datos de futuros
    df_futuros = get_data_futuros()
    df_futuros.to_csv(f"{DATA_PATH}/datos_futuros.csv", index=False)

    # Aquí puedes agregar más lógica para procesar o guardar los datos cargados
    print("Proceso completado.")
    return df_energia, df_futuros

if __name__ == "__main__":
    load_data_files()