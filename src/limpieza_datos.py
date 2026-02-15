#%% Importar librerías
import pandas as pd

#%% Configuración
class Config:
    HIERARCY_VERSIONS = {
            "TXF": "TX999999",
            "TX": "TX99999",
            "TXR": "TX9999",
            "TXA": "TX999"
        }

#%% Funciones de procesamiento de datos
def procesar_demanda(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Permite procesar el dataset de demanda, ordenando por fecha, código del agente,
    tipo de mercado y versión, y eliminando duplicados para quedarnos
    con la última versión de cada combinación única de fecha, código del agente y 
    tipo de mercado.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de demanda a procesar.
    
    Returns:
        pd.DataFrame: DataFrame procesado con los datos de demanda ordenados y sin duplicados.
    """

    # Convertir la columna "Fecha" a formato datetime
    df["FechaHora"] = pd.to_datetime(df["FechaHora"], format="%Y-%m-%d %H:%M:%S")

    # Extraer el número de versión de la columna "Version" para ordenar correctamente
    df["Version"] = df["Version"].replace(Config.HIERARCY_VERSIONS)
    df["v_num"] = df["Version"].str.extract("(\d+)").astype(int)

    # Ordenar por fecha, código del agente, tipo de mercado y versión
    df_sorted = df.sort_values(
        by=["FechaHora", "CodigoSICAgente", "TipoMercado", "v_num"],
        ascending=True
    )

    # Eliminar duplicados, quedándonos con la última versión de cada combinación
    # única de fecha, código del agente y tipo de mercado
    df_clean = df_sorted.drop_duplicates(
        subset=["FechaHora", "CodigoSICAgente", "TipoMercado"],
        keep="last"
    ).drop(columns=["v_num"])

    # Calcular la demanda diaria, y también por franjas horarios:
    # 0 a 7, 7 a 17, 17 a 24 y dia completo
    df_clean["Hora"] = df_clean["FechaHora"].dt.hour
    df_clean["FranjaHoraria"] = pd.cut(
        df_clean["Hora"],
        bins=[-1, 7, 16, 23],
        labels=["0-7", "7-17", "17-23"]
    )
    
    df_final1 = df_clean.groupby(
        [df_clean["FechaHora"].dt.date, "FranjaHoraria"]
    )["Valor"].sum().reset_index()

    df_final2 = df_clean.groupby(
        df_clean["FechaHora"].dt.date
    )["Valor"].sum().reset_index()
    df_final2["FranjaHoraria"] = "Dia"

    df_final = pd.concat([df_final1, df_final2], ignore_index=True)

    # Renombrar columnas para mayor claridad
    df_final.rename(columns={"FechaHora": "Fecha", "Valor": "Demanda_kWh"}, inplace=True)

    # Formato wide, con columnas para cada franja horaria
    df_final = df_final.pivot(
        index="Fecha", columns="FranjaHoraria",
        values="Demanda_kWh"
    ).reset_index()

    # Completar fechas faltantes con demanda promediado de los 30 días anteriores
    df_final["Fecha"] = pd.to_datetime(df_final["Fecha"])
    df_final.set_index("Fecha", inplace=True)
    df_final = df_final.asfreq("D")  # Asegura que todas las fechas estén presentes
    df_final = df_final.sort_index()  # Asegura que las fechas estén ordenadas
    for col in ["0-7", "7-17", "17-23", "Dia"]:
        df_final[col] = df_final[col].fillna(df_final[col].rolling(window=30, min_periods=1).mean())
    df_final = df_final.reset_index()  # Volver a tener "Fecha" como columna normal

    # Renombrar columnas para mayor claridad
    df_final.rename(
        columns={
            "0-7": "Demanda_kWh_0-7",
            "7-17": "Demanda_kWh_7-17",
            "17-23": "Demanda_kWh_17-23",
            "Dia": "Demanda_kWh_Dia"
        },
        inplace=True
    )

    df_final = df_final[
        [
            "Fecha", "Demanda_kWh_0-7", "Demanda_kWh_7-17",
            "Demanda_kWh_17-23", "Demanda_kWh_Dia"
        ]
    ]

    # Dataframe para comprador no regulado GECC
    df_comprador = df_clean[df_clean["CodigoSICAgente"] == "GECC"]

    df_comprador1 = df_comprador.groupby(
        [df_comprador["FechaHora"].dt.date, "FranjaHoraria"]
    )["Valor"].sum().reset_index()

    df_comprador2 = df_comprador.groupby(
        df_comprador["FechaHora"].dt.date
    )["Valor"].sum().reset_index()
    df_comprador2["FranjaHoraria"] = "Dia"

    df_comprador_final = pd.concat([df_comprador1, df_comprador2], ignore_index=True)

    # Renombrar columnas para mayor claridad
    df_comprador_final.rename(columns={"FechaHora": "Fecha", "Valor": "Demanda_kWh"}, inplace=True)

    # Formato wide, con columnas para cada franja horaria
    df_comprador_final_wide = df_comprador_final.pivot(
        index="Fecha", columns="FranjaHoraria",
        values="Demanda_kWh"
    ).reset_index()

    # Completar fechas faltantes con demanda promediado de los 30 días anteriores
    df_comprador_final_wide["Fecha"] = pd.to_datetime(df_comprador_final_wide["Fecha"])
    df_comprador_final_wide.set_index("Fecha", inplace=True)
    df_comprador_final_wide = df_comprador_final_wide.asfreq("D")  # Asegura que todas las fechas estén presentes
    df_comprador_final_wide = df_comprador_final_wide.sort_index()  # Asegura que las fechas estén ordenadas
    for col in ["0-7", "7-17", "17-23", "Dia"]:
        df_comprador_final_wide[col] = df_comprador_final_wide[col].fillna(df_comprador_final_wide[col].rolling(window=30, min_periods=1).mean())
    df_comprador_final_wide = df_comprador_final_wide.reset_index()  # Volver a tener "Fecha" como columna normal

    # Renombrar columnas para mayor claridad
    df_comprador_final_wide.rename(
        columns={
            "0-7": "Demanda_kWh_0-7",
            "7-17": "Demanda_kWh_7-17",
            "17-23": "Demanda_kWh_17-23",
            "Dia": "Demanda_kWh_Dia"
        },
        inplace=True
    )

    df_comprador_final = df_comprador_final_wide[
        [
            "Fecha", "Demanda_kWh_0-7", "Demanda_kWh_7-17",
            "Demanda_kWh_17-23", "Demanda_kWh_Dia"
        ]
    ]

    return (df_final, df_comprador_final)


def procesar_precios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permite procesar el dataset de precios, ordenando por fecha y versión,
    y eliminando duplicados para quedarnos con la última versión de cada fecha/hora.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de precios a procesar.
    
    Returns:
        pd.DataFrame: DataFrame procesado con los datos de precios ordenados y sin duplicados.
    """

    # Filtrar solo PB_Tie
    df = df[df["CodigoVariable"] == "PB_Tie"]

    # Convertir la columna "Fecha" a formato datetime
    df["FechaHora"] = pd.to_datetime(df["FechaHora"], format="%Y-%m-%d %H:%M:%S")

    # Extraer el número de versión de la columna "Version" para ordenar correctamente
    df["Version"] = df["Version"].replace(Config.HIERARCY_VERSIONS)
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

    # Calcular precio promedio diario, y también por franjas horarios:
    # 0 a 7, 7 a 17, 17 a 24 y dia completo
    df_clean["Hora"] = df_clean["FechaHora"].dt.hour
    df_clean["FranjaHoraria"] = pd.cut(
        df_clean["Hora"],
        bins=[-1, 7, 16, 23],
        labels=["0-7", "7-17", "17-23"]
    )
    
    df_final1 = df_clean.groupby(
        [df_clean["FechaHora"].dt.date, "FranjaHoraria"]
    )["Valor"].mean().reset_index()

    df_final2 = df_clean.groupby(
        df_clean["FechaHora"].dt.date
    )["Valor"].mean().reset_index()
    df_final2["FranjaHoraria"] = "Dia"

    df_final = pd.concat([df_final1, df_final2], ignore_index=True)

    # Renombrar columnas para mayor claridad
    df_final.rename(columns={"FechaHora": "Fecha", "Valor": "Precio_COP/kWh"}, inplace=True)

    # Formato wide, con columnas para cada franja horaria
    df_final = df_final.pivot(
        index="Fecha", columns="FranjaHoraria",
        values="Precio_COP/kWh"
    ).reset_index()

    # Completar fechas faltantes con precio promediado de los 30 días anteriores
    df_final["Fecha"] = pd.to_datetime(df_final["Fecha"])
    df_final.set_index("Fecha", inplace=True)
    df_final = df_final.asfreq("D")  # Asegura que todas las fechas estén presentes
    df_final = df_final.sort_index()  # Asegura que las fechas estén ordenadas
    for col in ["0-7", "7-17", "17-23", "Dia"]:
        df_final[col] = df_final[col].fillna(df_final[col].rolling(window=30, min_periods=1).mean())
    df_final = df_final.reset_index()  # Volver a tener "Fecha" como columna normal
    
    # Renombrar columnas para mayor claridad
    df_final.rename(
        columns={
            "0-7": "Precio_COP/kWh_0-7",
            "7-17": "Precio_COP/kWh_7-17",
            "17-23": "Precio_COP/kWh_17-23",
            "Dia": "Precio_COP/kWh_Dia"
        },
        inplace=True
    )

    df_final = df_final[
        [
            "Fecha", "Precio_COP/kWh_0-7", "Precio_COP/kWh_7-17",
            "Precio_COP/kWh_17-23", "Precio_COP/kWh_Dia"
        ]
    ]
    return df_final


def procesar_precios_ponderados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permite procesar el dataset de precios ponderados, ordenando por fecha, código del agente,
    tipo de mercado y versión, y eliminando duplicados para quedarnos
    con la última versión de cada fecha/hora
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de precios ponderados a procesar.
    
    Returns:
        pd.DataFrame: DataFrame procesado con los datos de precios ponderados ordenados y sin duplicados.
    """

    # Filtrar solo PB_Tie_Ponderado
    df = df[df["CodigoVariable"] == "PPBOGReal"]

    # Convertir la columna "Fecha" a formato datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%Y-%m-%d")

    # Extraer el número de versión de la columna "Version" para ordenar correctamente
    df["Version"] = df["Version"].replace(Config.HIERARCY_VERSIONS)
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


def procesar_aportes_hidricos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permite procesar el dataset de aportes hídricos, ordenando por fecha,
    y agregando los aportes hídricos por día.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de aportes hídricos a procesar.
    
    Returns:
        pd.DataFrame: DataFrame procesado con los datos de aportes hídricos ordenados y sin duplicados.
    """

    # Convertir la columna "Fecha" a formato datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%Y-%m-%d")

    # Agrupar por fecha y sumar los aportes hídricos
    df_final = df.groupby("Fecha")["AportesHidricosEnergia"].sum().reset_index()

    # Completar fechas faltantes con aporte hídrico promediado de los 30 días anteriores
    df_final["Fecha"] = pd.to_datetime(df_final["Fecha"])
    df_final.set_index("Fecha", inplace=True)
    df_final = df_final.asfreq("D")  # Asegura que todas las
    df_final = df_final.sort_index()  # Asegura que las fechas estén ordenadas
    df_final["AportesHidricosEnergia"] = df_final["AportesHidricosEnergia"].fillna(df_final["AportesHidricosEnergia"].rolling(window=30, min_periods=1).mean())
    df_final = df_final.reset_index()  # Volver a tener "Fecha" como columna normal

    # Renombrar columnas para mayor claridad
    df_final.rename(columns={"AportesHidricosEnergia": "AportesHidricos_kWh"}, inplace=True)

    df_final = df_final[["Fecha", "AportesHidricos_kWh"]]
    return df_final


def procesar_niveles_embalse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permite procesar el dataset de niveles de embalse, ordenando por fecha y versión.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de niveles de embalse a procesar.
    
    Returns:
        pd.DataFrame: DataFrame procesado con los datos de niveles de embalse ordenados y sin duplicados.
    """

    # Convertir la columna "Fecha" a formato datetime
    df["Fecha"] = pd.to_datetime(df["FechaInicio"], format="%Y-%m-%d")

    # Extraer el número de versión de la columna "Version" para ordenar correctamente
    df["Version"] = df["Version"].replace(Config.HIERARCY_VERSIONS)
    df["v_num"] = df["Version"].str.extract("(\d+)").astype(int)

    # Ordenar por fecha, código de la planta y versión
    df_sorted = df.sort_values(
        by=["Fecha", "CodigoPlanta", "v_num"],
        ascending=True
    )

    # Eliminar duplicados, quedándonos con la última versión de cada combinación
    # única de fecha y código de la planta
    df_clean = df_sorted.drop_duplicates(
        subset=["Fecha", "CodigoPlanta"],
        keep="last"
    ).drop(columns=["v_num"])

    # Agrupar por fecha y calcular el nivel de embalse total sumando los niveles de cada planta
    df_final = df_clean.groupby("Fecha")["Valor"].sum().reset_index()

    # Completar fechas faltantes con nivel de embalse promediado de los 30 días anteriores
    df_final["Fecha"] = pd.to_datetime(df_final["Fecha"])
    df_final.set_index("Fecha", inplace=True)
    df_final = df_final.asfreq("D")  # Asegura que todas las fechas estén presentes
    df_final = df_final.sort_index()  # Asegura que las fechas estén ordenadas
    df_final["Valor"] = df_final["Valor"].fillna(df_final["Valor"].rolling(window=30, min_periods=1).mean())
    df_final = df_final.reset_index()  # Volver a tener "Fecha" como columna normal

    # Renombrar columnas para mayor claridad
    df_final.rename(columns={"Valor": "NivelEmbalse"}, inplace=True)

    df_final = df_final[["Fecha", "NivelEmbalse"]]
    return df_final


def procesar_disponibilidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permite procesar el dataset de disponibilidad real, ordenando por fecha y versión.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de disponibilidad real a procesar.
    
    Returns:
        pd.DataFrame: DataFrame procesado con los datos de disponibilidad real ordenados y sin duplicados.
    """

    # Convertir la columna "FechaHora" a formato datetime
    df["FechaHora"] = pd.to_datetime(df["FechaHora"], format="%Y-%m-%d %H:%M:%S")

    # Extraer el número de versión de la columna "Version" para ordenar correctamente
    df["Version"] = df["Version"].replace(Config.HIERARCY_VERSIONS)
    df["v_num"] = df["Version"].str.extract("(\d+)").astype(int)

    # Ordenar por fecha, código de la planta y versión
    df_sorted = df.sort_values(
        by=["FechaHora", "CodigoPlanta", "v_num"],
        ascending=True
    )

    # Eliminar duplicados, quedándonos con la última versión de cada combinación
    # única de fecha y código de la planta
    df_clean = df_sorted.drop_duplicates(
        subset=["FechaHora", "CodigoPlanta"],
        keep="last"
    ).drop(columns=["v_num"])

    # Calcular la disponibilidad diaria, y también por franjas horarios:
    # 0 a 7, 7 a 17, 17 a 24 y dia completo
    df_clean["Hora"] = df_clean["FechaHora"].dt.hour
    df_clean["FranjaHoraria"] = pd.cut(
        df_clean["Hora"],
        bins=[-1, 7, 16, 23],
        labels=["0-7", "7-17", "17-23"]
    )
    
    df_final1 = df_clean.groupby(
        [df_clean["FechaHora"].dt.date, "FranjaHoraria"]
    )["Valor"].sum().reset_index()

    df_final2 = df_clean.groupby(
        df_clean["FechaHora"].dt.date
    )["Valor"].sum().reset_index()
    df_final2["FranjaHoraria"] = "Dia"

    df_final = pd.concat([df_final1, df_final2], ignore_index=True)

    # Renombrar columnas para mayor claridad
    df_final.rename(columns={"FechaHora": "Fecha", "Valor": "Disponibilidad_kWh"}, inplace=True)

    # Formato wide, con columnas para cada franja horaria
    df_final = df_final.pivot(
        index="Fecha", columns="FranjaHoraria",
        values="Disponibilidad_kWh"
    ).reset_index()

    # Completar fechas faltantes con disponibilidad promediado de los 30 días anteriores
    df_final["Fecha"] = pd.to_datetime(df_final["Fecha"])
    df_final.set_index("Fecha", inplace=True)
    df_final = df_final.asfreq("D")  # Asegura que todas las fechas estén presentes
    df_final = df_final.sort_index()  # Asegura que las fechas estén ordenadas
    for col in ["0-7", "7-17", "17-23", "Dia"]:
        df_final[col] = df_final[col].fillna(df_final[col].rolling(window=30, min_periods=1).mean())
    df_final = df_final.reset_index()  # Volver a tener "Fecha" como columna normal

    # Renombrar columnas para mayor claridad
    df_final.rename(
        columns={
            "0-7": "Disponibilidad_kWh_0-7",
            "7-17": "Disponibilidad_kWh_7-17",
            "17-23": "Disponibilidad_kWh_17-23",
            "Dia": "Disponibilidad_kWh_Dia"
        },
        inplace=True
    )

    df_final = df_final[
        [
            "Fecha", "Disponibilidad_kWh_0-7", "Disponibilidad_kWh_7-17",
            "Disponibilidad_kWh_17-23", "Disponibilidad_kWh_Dia"
        ]
    ]
    return df_final


def procesar_generacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permite procesar el dataset de generación real, ordenando por fecha.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de generación real a procesar.
    Returns:
        pd.DataFrame: DataFrame procesado con los datos de generación real ordenados y sin duplicados.
    """

    # Convertir la columna "Fecha" a formato datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%Y-%m-%d")

    # Agrupar por fecha y tipo de generación, y sumar la generación de cada tipo
    df_final1 = df.groupby(
        ["Fecha", "TipoGeneracion"]
    )["GeneracionRealEstimada"].sum().reset_index()

    df_final2 = df.groupby(
        "Fecha"
    )["GeneracionRealEstimada"].sum().reset_index()
    df_final2["TipoGeneracion"] = "Total"

    # Concatenar los dos dataframes
    df_final = pd.concat([df_final1, df_final2], ignore_index=True)

    # Pasar a formato wide, con columnas para cada tipo de generación
    df_final = df_final.pivot(
        index="Fecha", columns="TipoGeneracion",
        values="GeneracionRealEstimada"
    ).reset_index()

    # Completar fechas faltantes con generación promediada de los 30 días anteriores
    df_final["Fecha"] = pd.to_datetime(df_final["Fecha"])
    df_final.set_index("Fecha", inplace=True)
    df_final = df_final.asfreq("D")  # Asegura que todas las fechas estén presentes
    df_final = df_final.sort_index()  # Asegura que las fechas estén ordenadas
    columns_to_fill = [col for col in df_final.columns if col != "Fecha"]
    for col in columns_to_fill:
        df_final[col] = df_final[col].fillna(df_final[col].rolling(window=30, min_periods=1).mean())
    df_final = df_final.reset_index()  # Volver a tener "Fecha" como columna normal

    # Renombrar columnas para mayor claridad
    df_final.rename(columns={v: f"Generacion_{v}_kWh" for v in columns_to_fill}, inplace=True)

    df_final = df_final[["Fecha"] + [f"Generacion_{v}_kWh" for v in columns_to_fill]]

    return df_final