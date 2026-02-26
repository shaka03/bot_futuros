#%% Librerías

# Librerías úitiles
import os
import pandas as pd
import numpy as np

from utils.simulation_demand import run_simulation

import warnings
warnings.filterwarnings("ignore")


#%% Configuración
class Config:
    SILVER_DATA_PATH = os.path.join(os.getcwd(), "data/2_silver")
    GOLD_DATA_PATH = os.path.join(os.getcwd(), "data/3_gold")

    DICT_FILES = {
        "futuros": "precios_FUTUROS.csv",
        "futuros_wide": "precios_FUTUROS_WIDE.csv",
        "precios": "datos_PRECIOS.csv",
        "precios_ponderados": "datos_PRECIOS_PONDERADOS.csv",
        "precios_bilaterales": "datos_PRECIOS_BILATERALES.csv",
        "demanda": "datos_DEMANDA.csv",
        "demanda_comprador": "datos_DEMANDA_COMPRADOR.csv",
        "aportes_hidricos": "datos_APORTES_HIDRICOS.csv",
        #"niveles_embalse": "datos_NIVELES_EMBALSE.csv",
        "generacion": "datos_GENERACION_REAL.csv",
        "disponibilidad": "datos_DISPONIBILIDAD_REAL.csv",
        "noticias": "datos_NOTICIAS.csv"
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


def process_data(
        fecha_inicio_str: str,
        fecha_fin_str: str,
        fecha_inicio_transacciones_str: str
    ) -> None:
    print("Iniciando procesamiento de datos...")

    list_var_sistema = []
    for key, filename in Config.DICT_FILES.items():
        file_path = os.path.join(Config.SILVER_DATA_PATH, filename)
        if os.path.exists(file_path):
            print(f"Procesando datos {key} ...")

            if key == "demanda_comprador":
                df = pd.read_csv(file_path, parse_dates=["Fecha"])
                df_comprador = run_simulation(df.copy())
                df_comprador.to_csv(os.path.join(Config.GOLD_DATA_PATH, "datos_DEMANDA_COMPRADOR.csv"), index=False)
            
            elif key == "futuros":
                df = pd.read_csv(file_path, parse_dates=["Fecha"])
                df.to_csv(os.path.join(Config.GOLD_DATA_PATH, "precios_FUTUROS.csv"), index=False)

                # Guardar fechas transacciones
                df_fechas = df[df["Fecha"] >= fecha_inicio_transacciones_str][["Fecha"]]
                df_fechas = df_fechas.drop_duplicates().sort_values("Fecha").reset_index(drop=True)
                df_fechas.to_csv(os.path.join(Config.GOLD_DATA_PATH, "fechas_transacciones.csv"), index=False)
            
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
            
            elif key == "precios":
                df = pd.read_csv(file_path, parse_dates=["Fecha"])
                df.to_csv(os.path.join(Config.GOLD_DATA_PATH, "datos_PRECIOS.csv"), index=False)
                list_var_sistema.append(df)

                # Calcular los precios de liquidación por mes
                df_precios_mes = df.copy()
                df_precios_mes["FechaVencimiento"] = df_precios_mes["Fecha"].dt.to_period("M").dt.to_timestamp()
                df_precios_mes["FechaVencimiento"] =  df_precios_mes["FechaVencimiento"] + pd.offsets.MonthEnd(0) # Ajustar al último día del mes
                df_precios_mes = df_precios_mes.drop(columns=["Fecha"])
                df_precios_mes = df_precios_mes.groupby("FechaVencimiento").mean().reset_index()
                df_precios_mes.to_csv(os.path.join(Config.GOLD_DATA_PATH, "precios_LIQUIDACION.csv"), index=False)
            
            elif key == "noticias":
                df = pd.read_csv(file_path, parse_dates=["Fecha"])
                df = df[["Fecha", "Tipo_noticia"]]
                df = df.groupby("Fecha").mean().reset_index()

                # Completar fechas faltantes
                fechas_completas = pd.date_range(start=fecha_inicio_str, end=fecha_fin_str, freq="D")
                df = df.set_index("Fecha").reindex(fechas_completas).sort_index().ffill().rename_axis("Fecha").reset_index()
                # Aplicar media movil de los últimos 30 días
                df["Tipo_noticia"] = df["Tipo_noticia"].rolling(window=30, min_periods=1).mean()  # Suavizar con media móvil de 30 días
                list_var_sistema.append(df)
            
            elif key == "futuros_wide":
                df = pd.read_csv(file_path, parse_dates=["Fecha"])
                lista_meses = [
                    "Vencimiento_00Meses", "Vencimiento_01Meses", "Vencimiento_02Meses", "Vencimiento_03Meses",
                    "Vencimiento_04Meses", "Vencimiento_05Meses", "Vencimiento_06Meses"
                ]
                contratos = [
                    "ELM",
                    "MTB",
                    "DTB",
                    "NTB"
                ]
                cols_to_analyze = []
                for contrato in contratos:
                    for mes in lista_meses:
                        col_name = f"{contrato}_{mes}"
                        cols_to_analyze.append(col_name)
                df = df[["Fecha"] + cols_to_analyze]
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

        # Retornos precios spot
        df_sistema.set_index("Fecha", inplace=True)
        df_sistema.sort_index(inplace=True)
        df_sistema["Retorno_Precio_Dia"] = np.log(df_sistema["Precio_Ponderado_COP/kWh"] / df_sistema["Precio_Ponderado_COP/kWh"].shift(1))
        df_sistema["Retorno_Precio_0-7"] = np.log(df_sistema["Precio_COP/kWh_0-7"] / df_sistema["Precio_COP/kWh_0-7"].shift(1))
        df_sistema["Retorno_Precio_7-17"] = np.log(df_sistema["Precio_COP/kWh_7-17"] / df_sistema["Precio_COP/kWh_7-17"].shift(1))
        df_sistema["Retorno_Precio_17-23"] = np.log(df_sistema["Precio_COP/kWh_17-23"] / df_sistema["Precio_COP/kWh_17-23"].shift(1))

        # Calcular los beta móviles a 30 días para cada contrato de futuros
        print("Calculando beta móviles a 30 días y bases para precios futuros...")
        lista_meses = [
            "Vencimiento_00Meses", "Vencimiento_01Meses", "Vencimiento_02Meses", "Vencimiento_03Meses",
            "Vencimiento_04Meses", "Vencimiento_05Meses", "Vencimiento_06Meses"
        ]
        cols_to_analyze = {
            "ELM": [],
            "MTB": [],
            "DTB": [],
            "NTB": []
        }

        for tipo in cols_to_analyze.keys():
            cols_to_analyze[tipo] = [f"{tipo}_{mes}" for mes in lista_meses]
        
        for tipo, cols in cols_to_analyze.items():
            if tipo == "ELM":
                for col in cols:
                    df_sistema[col] = df_sistema[col].ffill().bfill()
                    # Cálculo de base
                    df_sistema[f"Base_{col}"] = df_sistema[col] - df_sistema["Precio_Ponderado_COP/kWh"]
                    # Cálculo de beta móvil 30 días
                    df_sistema[f"Retorno_{col}"] = np.log(df_sistema[col] / df_sistema[col].shift(1))
                    rolling_cov = df_sistema[f"Retorno_{col}"].rolling(window="30D", min_periods=30).cov(df_sistema["Retorno_Precio_Dia"])
                    rolling_var = df_sistema["Retorno_Precio_Dia"].rolling(window="30D", min_periods=30).var()
                    df_sistema[f"Beta_MA30_{col}"]  = rolling_cov / rolling_var
            if tipo == "MTB":
                for col in cols:
                    df_sistema[col] = df_sistema[col].ffill().bfill()
                    # Cálculo de base
                    df_sistema[f"Base_{col}"] = df_sistema[col] - df_sistema["Precio_COP/kWh_0-7"]
                    # Cálculo de beta móvil 30 días
                    df_sistema[f"Retorno_{col}"] = np.log(df_sistema[col] / df_sistema[col].shift(1))
                    rolling_cov = df_sistema[f"Retorno_{col}"].rolling(window="30D", min_periods=30).cov(df_sistema["Retorno_Precio_0-7"])
                    rolling_var = df_sistema["Retorno_Precio_0-7"].rolling(window="30D", min_periods=30).var()
                    df_sistema[f"Beta_MA30_{col}"]  = rolling_cov / rolling_var
            if tipo == "DTB":
                for col in cols:
                    df_sistema[col] = df_sistema[col].ffill().bfill()
                    # Cálculo de base
                    df_sistema[f"Base_{col}"] = df_sistema[col] - df_sistema["Precio_COP/kWh_7-17"]
                    # Cálculo de beta móvil 30 días
                    df_sistema[f"Retorno_{col}"] = np.log(df_sistema[col] / df_sistema[col].shift(1))
                    rolling_cov = df_sistema[f"Retorno_{col}"].rolling(window="30D", min_periods=30).cov(df_sistema["Retorno_Precio_7-17"])
                    rolling_var = df_sistema["Retorno_Precio_7-17"].rolling(window="30D", min_periods=30).var()
                    df_sistema[f"Beta_MA30_{col}"]  = rolling_cov / rolling_var
            if tipo == "NTB":
                for col in cols:
                    df_sistema[col] = df_sistema[col].ffill().bfill()
                    # Cálculo de base
                    df_sistema[f"Base_{col}"] = df_sistema[col] - df_sistema["Precio_COP/kWh_17-23"]
                    # Cálculo de beta móvil 30 días
                    df_sistema[f"Retorno_{col}"] = np.log(df_sistema[col] / df_sistema[col].shift(1))
                    rolling_cov = df_sistema[f"Retorno_{col}"].rolling(window="30D", min_periods=30).cov(df_sistema["Retorno_Precio_17-23"])
                    rolling_var = df_sistema["Retorno_Precio_17-23"].rolling(window="30D", min_periods=30).var()
                    df_sistema[f"Beta_MA30_{col}"]  = rolling_cov / rolling_var
        
        # Guardar los datasets finales en GOLD
        df_sistema = df_sistema.shift(1).reset_index()  # Desplazar para evitar usar información futura en el mismo día
        df_sistema.to_csv(os.path.join(Config.GOLD_DATA_PATH, "dataset_SISTEMA.csv"), index=False)
    
    print("Datasets finales guardado en GOLD.")
    return None


#%% Ejecutar carga de datos
if __name__ == "__main__":
    fecha_inicio_str = "2022-01-01"
    fecha_inicio_transacciones_str = "2022-02-01"
    fecha_fin_str = "2026-01-31"
    process_data(fecha_inicio_str, fecha_fin_str, fecha_inicio_transacciones_str)