#%% Importar librerías
import pandas as pd
import numpy as np

#%% Configuración
class Config:
    HIERARCY_VERSIONS = {
            "TXF": "TX999999",
            "TX": "TX99999",
            "TXR": "TX9999",
            "TXA": "TX999"
        }
    
    HIERARCY_VERSIONS_PRICE = {
            "TXR": "TX999999",
            "TXF": "TX99999",
            "TX": "TX9999",
            "TXA": "TX999"
        }

#%% Funciones de procesamiento de datos

def procesar_precios(
        df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Permite procesar el dataset de precios, ordenando por fecha y versión,
    y eliminando duplicados para quedarnos con la última versión de cada fecha/hora.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de precios a procesar.
        clean_outliers (bool): Indica si se deben limpiar los outliers.
    
    Returns:
        pd.DataFrame: DataFrame procesado con los datos de precios ordenados y sin duplicados.
    """

    # Filtrar solo PB_Nac
    df = df[df["CodigoVariable"] == "PB_Nal"]

    # Convertir la columna "FechaHora" a formato datetime
    df["FechaHora"] = pd.to_datetime(df["FechaHora"], format="%Y-%m-%d %H:%M:%S")

    # Extraer el número de versión de la columna "Version" para ordenar correctamente
    df["Version"] = df["Version"].replace(Config.HIERARCY_VERSIONS_PRICE)
    df["v_num"] = df["Version"].str.extract("(\d+)").astype(int)

    # Ordenar por fecha y versión
    df_sorted = df.sort_values(
        by=["FechaHora", "v_num"],
        ascending=True
    )

    # Eliminar duplicados, quedándonos con la última versión de cada combinación
    # única de fecha
    df_clean = df_sorted.drop_duplicates(
        subset=["FechaHora"],
        keep="last"
    ).drop(columns=["v_num"])

    # Obetner franjas horarias 0-7, 7-17 y 17-24
    df_clean["Hora"] = df_clean["FechaHora"].dt.hour
    df_clean["Franja"] = np.where(
        df_clean["Hora"] < 7, "0-7",
        np.where(
            df_clean["Hora"] < 17,
            "7-17",
            "17-24"
        )
    )

    # Agrupar por fecha y obtener el precio promedio de cada día
    df_grouped1 = df_clean.groupby(
        [df_clean["FechaHora"].dt.date, df_clean["Franja"]]
    )["Valor"].mean().reset_index()

    df_grouped2 = df_clean.groupby(
        df_clean["FechaHora"].dt.date
    )["Valor"].mean().reset_index()
    df_grouped2["Franja"] = "Dia"

    df_final = pd.concat([df_grouped1, df_grouped2], axis=0).reset_index(drop=True)

    # Cambiar formato long por wide
    df_final = df_final.pivot(index="FechaHora", columns="Franja", values="Valor").reset_index()

    # Renombrar columnas para mayor claridad
    df_final.rename(
        columns={
            "FechaHora": "Fecha",
            "0-7": "Precio_0-7_COP/kWh",
            "7-17": "Precio_7-17_COP/kWh",
            "17-24": "Precio_17-24_COP/kWh",
            "Dia": "Precio_Dia_COP/kWh"
        },
        inplace=True
    )

    # Completar fechas faltantes con precio promediado de los 30 días anteriores
    df_final["Fecha"] = pd.to_datetime(df_final["Fecha"])
    df_final.set_index("Fecha", inplace=True)
    df_final = df_final.asfreq("D")  # Asegura que todas las fechas estén presentes
    df_final = df_final.sort_index()  # Asegura que las fechas estén ordenadas
    for col in ["Precio_0-7_COP/kWh", "Precio_7-17_COP/kWh", "Precio_17-24_COP/kWh", "Precio_Dia_COP/kWh"]:
        df_final[col] = df_final[col].fillna(df_final[col].rolling(window=30, min_periods=1).mean())
    df_final = df_final.reset_index()  # Volver a tener "Fecha" como columna normal

    df_final = df_final[["Fecha", "Precio_0-7_COP/kWh", "Precio_7-17_COP/kWh", "Precio_17-24_COP/kWh", "Precio_Dia_COP/kWh"]]

    return df_final


def procesar_precios_ponderados(
        df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Permite procesar el dataset de precios ponderados, ordenando por fecha, código del agente,
    tipo de mercado y versión, y eliminando duplicados para quedarnos
    con la última versión de cada fecha/hora
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de precios ponderados a procesar.
        clean_outliers (bool): Indica si se deben limpiar los outliers.
    
    Returns:
        pd.DataFrame: DataFrame procesado con los datos de precios ponderados ordenados y sin duplicados.
    """

    # Filtrar solo PB_Tie_Ponderado
    df = df[df["CodigoVariable"] == "PPBOGReal"]

    # Convertir la columna "Fecha" a formato datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%Y-%m-%d")

    # Extraer el número de versión de la columna "Version" para ordenar correctamente
    df["Version"] = df["Version"].replace(Config.HIERARCY_VERSIONS_PRICE)
    df["v_num"] = df["Version"].str.extract("(\d+)").astype(int)

    # Ordenar por fecha y versión
    df_sorted = df.sort_values(
        by=["Fecha", "v_num"],
        ascending=True
    )

    # Eliminar duplicados, quedándonos con la última versión de cada combinación
    # única de fecha, código del agente y tipo de mercado
    df_clean = df_sorted.drop_duplicates(
        subset=["Fecha"],
        keep="last"
    ).drop(columns=["v_num"])

    # Renombrar columnas para mayor claridad
    df_final = df_clean.rename(columns={"Valor": "Precio_Ponderado_COP/kWh"})

    # Completar fechas faltantes con precio promediado de los 30 días anteriores
    df_final["Fecha"] = pd.to_datetime(df_final["Fecha"])
    df_final.set_index("Fecha", inplace=True)
    df_final = df_final.asfreq("D")  # Asegura que todas las fechas estén presentes
    df_final = df_final.sort_index()  # Asegura que las fechas estén ordenadas
    df_final["Precio_Ponderado_COP/kWh"] = df_final["Precio_Ponderado_COP/kWh"].fillna(df_final["Precio_Ponderado_COP/kWh"].rolling(window=30, min_periods=1).mean())
    df_final = df_final.reset_index()  # Volver a tener "Fecha" como columna normal

    df_final = df_final[["Fecha", "Precio_Ponderado_COP/kWh"]]

    return df_final


def procesar_bilaterales(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Permite procesar el dataset de precios bilaterales, ordenando por fecha y versión.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de precios bilaterales a procesar.
    Returns:
        pd.DataFrame: DataFrame procesado con los datos de precios bilaterales ordenados y sin duplicados.
    """
    # Convertir la columna "Fecha" a formato datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%Y-%m-%d")

    # Extraer el número de versión de la columna "Version" para ordenar correctamente
    df["Version"] = df["Version"].replace(Config.HIERARCY_VERSIONS_PRICE)
    df["v_num"] = df["Version"].str.extract("(\d+)").astype(int)

    # Ordenar por fecha y versión
    df_sorted = df.sort_values(
        by=["Fecha", "Hora", "v_num"],
        ascending=True
    )

    # Eliminar duplicados, quedándonos con la última versión de cada combinación
    # única de fecha
    df_clean = df_sorted.drop_duplicates(
        subset=["Fecha", "Hora"],
        keep="last"
    ).drop(columns=["v_num"])

    # Obtener franjas horarias 0-7, 7-17 y 17-24
    df_clean["Franja"] = np.where(
        df_clean["Hora"] < 7, "0-7",
        np.where(
            df_clean["Hora"] < 17,
            "7-17",
            "17-24"
        )
    )

    # Agrupar por franja horaria y fecha, y calcula el precio promedio de cada día
    df_clean["Fecha"] = df_clean["Fecha"].dt.date

    df_grouped1 = df_clean.groupby(
        [df_clean["Fecha"], df_clean["Franja"]]
    )["PPP"].mean().reset_index()

    df_grouped2 = df_clean.groupby(
        df_clean["Fecha"]
    )["PPP"].mean().reset_index()
    df_grouped2["Franja"] = "Dia"

    df_final = pd.concat([df_grouped1, df_grouped2], axis=0).reset_index(drop=True)

    # Cambiar formato long por wide
    df_final = df_final.pivot(index="Fecha", columns="Franja", values="PPP").reset_index()

    # Renombrar columnas para mayor claridad
    df_final.rename(
        columns={
            "Fecha": "Fecha",
            "0-7": "Precio_Bilateral_0-7_COP/kWh",
            "7-17": "Precio_Bilateral_7-17_COP/kWh",
            "17-24": "Precio_Bilateral_17-24_COP/kWh",
            "Dia": "Precio_Bilateral_Dia_COP/kWh"
        },
        inplace=True
    )

    # Completar fechas faltantes con precio promediado de los 30 días anteriores
    df_final["Fecha"] = pd.to_datetime(df_final["Fecha"])
    df_final.set_index("Fecha", inplace=True)
    df_final = df_final.asfreq("D")  # Asegura que todas las fechas estén presentes
    df_final = df_final.sort_index()  # Asegura que las fechas estén ordenadas
    for col in ["Precio_Bilateral_0-7_COP/kWh", "Precio_Bilateral_7-17_COP/kWh", "Precio_Bilateral_17-24_COP/kWh", "Precio_Bilateral_Dia_COP/kWh"]:
        df_final[col] = df_final[col].ffill()
    df_final = df_final.reset_index()  # Volver a tener "Fecha" como columna normal

    df_final = df_final[["Fecha", "Precio_Bilateral_0-7_COP/kWh", "Precio_Bilateral_7-17_COP/kWh", "Precio_Bilateral_17-24_COP/kWh", "Precio_Bilateral_Dia_COP/kWh"]]

    return df_final