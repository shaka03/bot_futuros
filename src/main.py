#%% Librerías

# Librerías úitiles
import os
import pandas as pd
import datetime as dt

# Librerías para cargar datos
from load_data_energia import load_data as get_data_energia
from load_data_futuros import get_data_futuros

#%% Ejecución
if __name__ == "__main__":
    print("Iniciando carga de datos...")
    # Cargar datos de energía
    df_energia = get_data_energia(
        fecha_inicio_str="2024-01-01",
        fecha_fin_str="2024-06-30"
    )
    
    # Cargar datos de futuros
    df_futuros = get_data_futuros()

    # Aquí puedes agregar más lógica para procesar o guardar los datos cargados
    print("Proceso completado.")