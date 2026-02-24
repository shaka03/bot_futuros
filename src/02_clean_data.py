#%% Librerías

# Librerías úitiles
import os
import pandas as pd
import numpy as np

from utils.simulation_demand import run_simulation


#%% Configuración
class Config:
    SILVER_DATA_PATH = os.path.join(os.getcwd(), "data/2_silver")
    GOLD_DATA_PATH = os.path.join(os.getcwd(), "data/3_gold")

    DICT_FILES = {
        "futuros": "precios_FUTUROS.csv",
        "precios": "datos_PRECIOS.csv",
        "precios_ponderados": "datos_PRECIOS_PONDERADOS.csv",
        #"precios_bilaterales": "datos_PRECIOS_BILATERALES.csv",
        "demanda": "datos_DEMANDA.csv",
        "demanda_comprador": "datos_DEMANDA_COMPRADOR.csv",
        "aportes_hidricos": "datos_APORTES_HIDRICOS.csv",
        #"niveles_embalse": "datos_NIVELES_EMBALSE.csv",
        "generacion": "datos_GENERACION_REAL.csv",
        "disponibilidad": "datos_DISPONIBILIDAD_REAL.csv",
        #"noticias": "datos_NOTICIAS.csv"
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

                # Completar fechas con el ultimo precio conocido (forward fill)
                fechas_completas = pd.DataFrame({"FechaCompleta": pd.date_range(start=fecha_inicio_str, end=fecha_fin_str, freq="D")})
                list_df = []
                for contrato in df["Nemotecnico"].unique():
                    # Filtrar por contrato
                    mask_contrato = df["Nemotecnico"] == contrato
                    df_contrato = df[mask_contrato].copy()

                    # Completar fechas faltantes con forward fill
                    df_contrato = pd.merge(fechas_completas, df_contrato, left_on="FechaCompleta", right_on="Fecha", how="left")
                    df_contrato.sort_values("FechaCompleta", inplace=True)
                    df_contrato.ffill(inplace=True)
                    cols_bfill = ["Nemotecnico", "Tipo", "Mes", "Año", "FechaVencimientoContrato"]
                    df_contrato[cols_bfill] = df_contrato[cols_bfill].bfill()
                    df_contrato.drop(columns=["Fecha"], inplace=True)
                    df_contrato.rename(columns={"FechaCompleta": "Fecha"}, inplace=True)
                    list_df.append(df_contrato)
                    del df_contrato
                
                # Concatenar todos los contratos
                df_futuros = pd.concat(list_df, ignore_index=True)    
            
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

        # Calcular los beta móviles a 30 días para cada contrato de futuros
        print("Calculando beta móviles a 30 días para cada contrato de futuros...")
        list_df_futuros = []
        for contrato in df_futuros["Nemotecnico"].unique():
            # Filtrar por contrato
            mask_contrato = df_futuros["Nemotecnico"] == contrato
            df_contrato = df_futuros[mask_contrato].copy()
            tipo_contrato = df_contrato["Tipo"].iloc[0]

            if len(df_contrato) < 30:
                print(f"  No hay suficientes datos para calcular beta móvil de 30 días para el contrato {contrato}. Se requiere al menos 30 registros.")
                list_df_futuros.append(df_contrato)
                del df_contrato
            else:
                df_contrato["Retorno_Futuros"] = np.log(df_contrato["Precio"] / df_contrato["Precio"].shift(1))
                if tipo_contrato == "ELM":
                    # Unir con el sistema para tener los precios spot
                    df_contrato = pd.merge(df_contrato, df_sistema[["Fecha", "Precio_Ponderado_COP/kWh"]], on="Fecha", how="left")
                    df_contrato.set_index("Fecha", inplace=True)
                    df_contrato.sort_index(inplace=True)
                    df_contrato["Base_Precio"] = df_contrato["Precio"] - df_contrato["Precio_Ponderado_COP/kWh"]

                    # Calcular retornos logarítmicos precio spot
                    df_contrato["Retorno_Precio"] = np.log(df_contrato["Precio_Ponderado_COP/kWh"] / df_contrato["Precio_Ponderado_COP/kWh"].shift(1))
                if tipo_contrato == "MTB":
                    # Unir con el sistema para tener los precios spot
                    df_contrato = pd.merge(df_contrato, df_sistema[["Fecha", "Precio_COP/kWh_0-7"]], on="Fecha", how="left")
                    df_contrato.set_index("Fecha", inplace=True)
                    df_contrato.sort_index(inplace=True)
                    df_contrato["Base_Precio"] = df_contrato["Precio"] - df_contrato["Precio_COP/kWh_0-7"]

                    # Calcular retornos logarítmicos precio spot
                    df_contrato["Retorno_Precio"] = np.log(df_contrato["Precio_COP/kWh_0-7"] / df_contrato["Precio_COP/kWh_0-7"].shift(1))

                if tipo_contrato == "DTB":
                    # Unir con el sistema para tener los precios spot
                    df_contrato = pd.merge(df_contrato, df_sistema[["Fecha", "Precio_COP/kWh_7-17"]], on="Fecha", how="left")
                    df_contrato.set_index("Fecha", inplace=True)
                    df_contrato.sort_index(inplace=True)
                    df_contrato["Base_Precio"] = df_contrato["Precio"] - df_contrato["Precio_COP/kWh_7-17"]

                    # Calcular retornos logarítmicos precio spot
                    df_contrato["Retorno_Precio"] = np.log(df_contrato["Precio_COP/kWh_7-17"] / df_contrato["Precio_COP/kWh_7-17"].shift(1))
                
                if tipo_contrato == "NTB":
                    # Unir con el sistema para tener los precios spot
                    df_contrato = pd.merge(df_contrato, df_sistema[["Fecha", "Precio_COP/kWh_17-23"]], on="Fecha", how="left")
                    df_contrato.set_index("Fecha", inplace=True)
                    df_contrato.sort_index(inplace=True, ascending=True)
                    df_contrato["Base_Precio"] = df_contrato["Precio"] - df_contrato["Precio_COP/kWh_17-23"]

                    # Calcular retornos logarítmicos precio spot
                    df_contrato["Retorno_Precio"] = np.log(df_contrato["Precio_COP/kWh_17-23"] / df_contrato["Precio_COP/kWh_17-23"].shift(1))
                
                # Calcular beta móvil de 30 días
                rolling_cov = df_contrato["Retorno_Futuros"].rolling(window="30D", min_periods=30).cov(df_contrato["Retorno_Precio"])
                rolling_var = df_contrato["Retorno_Precio"].rolling(window="30D", min_periods=30).var()
                df_contrato[f"Beta_Futuros_30D"] = rolling_cov / rolling_var
                df_contrato = df_contrato.shift(1)  # Desplazar beta para que corresponda al día actual (no usar información futura)
                df_contrato = df_contrato.reset_index()

                list_df_futuros.append(df_contrato)
                del df_contrato
        
        # Retornos precios spot
        df_sistema.set_index("Fecha", inplace=True)
        df_sistema.sort_index(inplace=True)
        df_sistema["Retorno_Precio_Dia"] = np.log(df_sistema["Precio_Ponderado_COP/kWh"] / df_sistema["Precio_Ponderado_COP/kWh"].shift(1))
        df_sistema["Retorno_Precio_0-7"] = np.log(df_sistema["Precio_COP/kWh_0-7"] / df_sistema["Precio_COP/kWh_0-7"].shift(1))
        df_sistema["Retorno_Precio_7-17"] = np.log(df_sistema["Precio_COP/kWh_7-17"] / df_sistema["Precio_COP/kWh_7-17"].shift(1))
        df_sistema["Retorno_Precio_17-23"] = np.log(df_sistema["Precio_COP/kWh_17-23"] / df_sistema["Precio_COP/kWh_17-23"].shift(1))
        
        # Guardar los datasets finales en GOLD
        ## Guardar sistema con variables calculadas
        df_sistema = df_sistema.shift(1).reset_index()  # Desplazar para evitar usar información futura en el mismo día
        df_sistema.to_csv(os.path.join(Config.GOLD_DATA_PATH, "dataset_SISTEMA.csv"), index=False)
        ## Guardar futuros con variables calculadas
        df_futuros_final = pd.concat(list_df_futuros, ignore_index=True)
        df_futuros_final = df_futuros_final[["Fecha", "Nemotecnico", "Tipo", "Precio", "Retorno_Futuros", "Beta_Futuros_30D", "Base_Precio"]]
        df_futuros_final.to_csv(os.path.join(Config.GOLD_DATA_PATH, "datos_FUTUROS.csv"), index=False)
    
    print("Datasets finales guardado en GOLD.")
    return None


#%% Ejecutar carga de datos
if __name__ == "__main__":
    fecha_inicio_str = "2022-01-01"
    fecha_inicio_transacciones_str = "2022-02-01"
    fecha_fin_str = "2026-01-31"
    process_data(fecha_inicio_str, fecha_fin_str, fecha_inicio_transacciones_str)