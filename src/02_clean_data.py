#%% Librerías

# Librerías úitiles
import os
import pandas as pd
import numpy as np


#%% Configuración
class Config:
    SILVER_DATA_PATH = os.path.join(os.getcwd(), "data/2_silver")
    GOLD_DATA_PATH = os.path.join(os.getcwd(), "data/3_gold")

    DICT_FILES = {
        "futuros": "precios_FUTUROS.csv",
        "precios": "datos_PRECIOS.csv",
        "precios_ponderados": "datos_PRECIOS_PONDERADOS.csv",
        "demanda": "datos_DEMANDA.csv",
        "demanda_comprador": "datos_DEMANDA_COMPRADOR.csv",
        "aportes_hidricos": "datos_APORTES_HIDRICOS.csv",
        "niveles_embalse": "datos_NIVELES_EMBALSE.csv",
        "generacion": "datos_GENERACION_REAL.csv",
        "disponibilidad": "datos_DISPONIBILIDAD_REAL.csv",
    }

#%% Funciones

def limpiar_outliers(
        df: pd.DataFrame,
        col: str
) -> pd.DataFrame:
    
    print(f"  Limpiando outliers en columna: {col}...")

    # 1. Calcular la Mediana Móvil (Referencia Robusta)
    # Usamos una ventana de 7 días porque la electricidad tiene ciclos semanales
    df = df.set_index("Fecha")
    mediana_movil = df[col].rolling(window=7, center=True).median()

    # 2. Calcular el Residuo (Diferencia entre dato real y la mediana)
    residuo = df[col] - mediana_movil

    # 3. Definir el Umbral de Anomalía (Estadístico)
    # Usamos el Rango Intercuartílico (IQR) o Desviación Estándar Robusta
    desviacion = residuo.std()
    umbral = 3 * desviacion  # 3 sigmas es estándar en industria

    # Identificar los índices de las anomalías
    is_outlier = abs(residuo) > umbral
    anomalias = df[is_outlier]
    
    print(f"  Anomalías detectadas: {len(anomalias)}")

    # 4. CORRECCIÓN: Reemplazar por Interpolación
    # Primero marcamos como NaN (nulo)
    df.loc[is_outlier, col] = np.nan

    # Luego interpolamos (rellenamos el hueco linealmente)
    df[col] = df[col].interpolate(method="time")
    df = df.reset_index()

    return df

def process_data():
    print("Iniciando procesamiento de datos...")

    list_var_sistema = []
    for key, filename in Config.DICT_FILES.items():
        file_path = os.path.join(Config.SILVER_DATA_PATH, filename)
        if os.path.exists(file_path):
            print(f"Cargando dataset {key} ...")
            
            if key in ["futuros", "demanda_comprador"]:
                df = pd.read_csv(file_path, parse_dates=["Fecha"])
                if key == "demanda_comprador":
                    df.to_csv(os.path.join(Config.GOLD_DATA_PATH, "datos_DEMANDA_COMPRADOR.csv"), index=False)
                if key == "futuros":
                    df.to_csv(os.path.join(Config.GOLD_DATA_PATH, "precios_FUTUROS.csv"), index=False)
            elif key == "niveles_embalse":
                df = pd.read_csv(file_path, parse_dates=["Fecha"])
                df = limpiar_outliers(df.copy(), "NivelEmbalse")
                list_var_sistema.append(df)
            elif key == "generacion":
                df = pd.read_csv(file_path, parse_dates=["Fecha"])
                cols_keep = [
                    "Fecha",
                    "Generacion_Termica_kWh",
                    "Generacion_Hidraulica_kWh"
                ]
                df = df[cols_keep]
                list_var_sistema.append(df)
            else:
                df = pd.read_csv(file_path, parse_dates=["Fecha"])
                list_var_sistema.append(df)
            
        else:
            print(f"Archivo {file_path} no encontrado. Verifica la ruta y el nombre del archivo.")
    
    # Hacer merge de las variables del sistema
    if list_var_sistema:
        df_sistema = list_var_sistema[0].copy()
        for df in list_var_sistema[1:]:
            df_sistema = pd.merge(df_sistema, df, on="Fecha", how="outer")
    
        # Crear variable ratio de cobertura
        df_sistema["Ratio_Cobertura_Dia"] = df_sistema["Disponibilidad_kWh_Dia"] / df_sistema["Demanda_kWh_Dia"]
        df_sistema["Ratio_Cobertura_0-7"] = df_sistema["Disponibilidad_kWh_0-7"] / df_sistema["Demanda_kWh_0-7"]
        df_sistema["Ratio_Cobertura_7-17"] = df_sistema["Disponibilidad_kWh_7-17"] / df_sistema["Demanda_kWh_7-17"]
        df_sistema["Ratio_Cobertura_17-23"] = df_sistema["Disponibilidad_kWh_17-23"] / df_sistema["Demanda_kWh_17-23"]

        # Media móvil de 7 días en aportes hídricos
        df_sistema["AportesHidricos_GWh_MA7"] = df_sistema["AportesHidricos_GWh"].rolling(window=7, min_periods=1).mean()
        
        # Guardar el dataset final en GOLD
        df_sistema.to_csv(os.path.join(Config.GOLD_DATA_PATH, "dataset_SISTEMA.csv"), index=False)
        print("Dataset final guardado en GOLD.")

    # Cargar datos noticias
    
    print("Datasets finales guardado en GOLD.")
    return None


#%% Ejecutar carga de datos
if __name__ == "__main__":
    process_data()