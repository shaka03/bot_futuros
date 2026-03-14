#%% Librerías
import os
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#%% Parámetros
class Config:
    # Ruta de los datos
    DATA_PATH = os.path.join(os.getcwd(), "data/1_raw")

    # Diccionario de meses
    MAP_MESES = {
        "F": "Enero",
        "G": "Febrero",
        "H": "Marzo",
        "J": "Abril",
        "K": "Mayo",
        "M": "Junio",
        "N": "Julio",
        "Q": "Agosto",
        "U": "Septiembre",
        "V": "Octubre",
        "X": "Noviembre",
        "Z": "Diciembre"
    }

    # Lista contratos
    LISTA_CONTRATOS = [
        "ELM",
        "MTB",
        "DTB",
        "NTB"
    ]
#%% Funciones

## Precios cierre
def transformar_precio_cierre(df):
    """
    Transforma el DataFrame de precios de cierre desde formato ancho a formato largo,
    y extrae la información de Tipo, Mes, Año y Fecha.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos de precios de cierre en formato ancho.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame transformado en formato largo con columnas adicionales.
    """
    # Asegurar que Nemo está como fecha
    df["Nemo"] = pd.to_datetime(df["Nemo"], dayfirst=True)

    # Formato largo
    df_long = df.melt(
        id_vars="Nemo",
        var_name="Contrato",
        value_name="Precio"
    )

    # Tipo = primeras 3 letras
    df_long["Tipo"] = df_long["Contrato"].str[:3]

    # Mes = letra 4 convertida
    df_long["Mes"] = df_long["Contrato"].str[3].map(Config.MAP_MESES)

    # Año = posiciones 5-6 convertidas a 20xx
    df_long["Año"] = df_long["Contrato"].str[4:6].astype(int).apply(lambda x: 2000 + x)

    # Fecha = Nemo original
    df_long["Fecha"] = df_long["Nemo"]

    # Validación fecha y precio. Ejemplo: si la fecha es de febrero de 2024 y hay precio de enero 2024, eliminar ese registro
    df_long["Fecha"] = pd.to_datetime(df_long["Fecha"], errors="coerce")
    df_long["Fecha2"] = pd.to_datetime(df_long["Año"].astype(str) + "-" + df_long["Mes"].map(
        {
            "Enero": "01",
            "Febrero": "02",
            "Marzo": "03",
            "Abril": "04",
            "Mayo": "05",
            "Junio": "06",
            "Julio": "07",
            "Agosto": "08",
            "Septiembre": "09",
            "Octubre": "10",
            "Noviembre": "11",
            "Diciembre": "12"
        }
    ) + "-01", errors="coerce")
    df_long["FechaVencimientoContrato"] = df_long["Fecha2"] + pd.offsets.MonthEnd(0)
    df_long = df_long[(df_long["Fecha"] <= df_long["FechaVencimientoContrato"])]
    df_long = df_long.dropna(subset=["Precio"])

    # Reordenar columnas
    df_final = df_long[["Contrato", "Tipo", "Mes", "Año", "Fecha", "FechaVencimientoContrato", "Precio"]]
    
    # Cambiar nombre
    df_final.rename(columns={"Contrato": "Nemotecnico"}, inplace=True)

    # Asegurar que Precio es numérico
    df_final["Precio"] = pd.to_numeric(df_final["Precio"])

    return df_final


def cargar_datos_futuros(nombre_archivo: str) -> pd.DataFrame:
    """
    Carga y organiza los datos de precios de cierre de futuros desde archivos Excel.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame con los datos organizados de precios de cierre de futuros.
    """
    # Precios cierre futuros
    df_cierre_list = []
    for con in Config.LISTA_CONTRATOS:
        try:
            df = pd.read_excel(
                os.path.join(Config.DATA_PATH, nombre_archivo),
                sheet_name=f"Precio cierre {con}",
                skiprows=1
            )
        except Exception as e:
            print("No existe la hoja para el contrato", con, "en el archivo", nombre_archivo)
            continue
        # Mantener solo columnas con datos
        cols_keep = [x for x in df.columns if not x.startswith("Unnamed")]
        df = df[cols_keep]
        df_cierre_list.append(transformar_precio_cierre(df))
    df_cierre_futuros = pd.concat(df_cierre_list, ignore_index=True)

    # Obtener ELS
    #df_els = df_cierre_futuros[df_cierre_futuros["Tipo"] == "ELM"]
    #df_els["Tipo"] = "ELS"
    #df_els["Nemotecnico"] = df_els["Nemotecnico"].str.replace("ELM", "ELS", regex=False)
    #df_cierre_futuros = pd.concat([df_cierre_futuros, df_els], ignore_index=True)
    
    return df_cierre_futuros


## Convocatorias
def cargar_datos_convocatorias():
    """
    Carga y organiza los datos de convocatorias y precios de referencia desde archivos Excel.

    Retorna
    -------
    tuple of pd.DataFrame
        DataFrames con los datos organizados de convocatorias y precios de referencia.
    """
    # Cargar datos
    conv = pd.read_excel(os.path.join(Config.DATA_PATH, "Convocatorias.xlsx"), sheet_name="Resultados Convocatorias",skiprows=12,usecols="D:J")
    refe = pd.read_excel(os.path.join(Config.DATA_PATH, "Convocatorias.xlsx"), sheet_name="Precios de referencia",skiprows=4,usecols="D:G")

    # Organizar Convocatorias
    conv = conv.rename(
        columns={
            "Fecha Convocatoria": "Fecha",
            "Contrato convocado": "Convocado",
            "Precio Adjudicado $/kWh": "Precio_Adjudicado",
            "Cantidad de Energía Adjudicada GWh": "Energía_Adjudicada",
            "Cantidad de Energía Expuesta en Compra GWh":"Expuesta_Compra",
            "Cantidad de Energía Expuesta en Venta GWh":"Expuesta_Venta"
        }
    )
    conv = conv[~conv["Nemotecnico"].str.startswith("ELB")]
    conv["Convocado"] = pd.to_datetime(conv["Convocado"], dayfirst=True, errors="coerce")

    ## Extraer Tipo, Mes (letra) y Año (2 dígitos) desde el nemotécnico de la columna "Contrato"
    conv[["Tipo", "MesLetra", "Año2d"]] = conv["Nemotecnico"].str.extract(
        r"([A-Z]{3})([A-Z])(\d{2})"
    )

    ## Convertir la letra del mes a número
    conv["Mes"] = conv["MesLetra"].map(Config.MAP_MESES)

    ## Convertir año de 2 dígitos a formato completo
    conv["Año"] = conv["Año2d"].astype(int) + 2000

    ## Eliminar temporales
    conv = conv.drop(columns=["MesLetra", "Año2d"])

    ## Columnas nuevas
    nuevas = ["Tipo", "Mes", "Año"]

    ## Lista actual de columnas
    cols = list(conv.columns)

    ## Posición donde está la columna "Convocado"
    idx = cols.index("Convocado")

    ## Construir nuevo orden:
    ##   - todas las columnas antes de Convocado
    ##   - Convocado
    ##   - nuevas columnas
    ##   - el resto
    nuevo_orden = (
        cols[:idx+1] +
        nuevas +
        [c for c in cols if c not in nuevas and c not in cols[:idx+1]]
    )

    ## Reordenar dataframe
    conv = conv[nuevo_orden]

    # Organizar referencia
    refe = refe.rename(columns={
        "Nemotécnico": "Nemotecnico",
        "Precio $/kWh": "Precio"
    })
    refe = refe.drop(columns=["Contrato"])

    ## Año → se toma la posición 5 y 6, y se convierte a 20XX
    refe["Año"] = "20" + refe["Nemotecnico"].str[4:6]
    refe["Año"] = refe["Año"].astype(int)
    ## Mes → se toma la 4ta letra del nemotécnico
    refe["Mes"] =refe["Nemotecnico"].str[3].map(Config.MAP_MESES)

    return conv, refe


## Negociación electrónica
def transformar_neg(df):
    """
    Transforma el DataFrame de negociación electrónica para extraer información adicional.
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos de negociación electrónica.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame transformado con columnas adicionales.
    """
    # Asegurar que FECHA esté como datetime
    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True)

    # Renombrar CONTRATO → Contrato para consistencia
    df = df.rename(columns={"CONTRATO": "Contrato"})

    # Tipo = primeras 3 letras
    df["Tipo"] = df["Contrato"].str[:3]

    # Mes = 4ta letra convertida
    df["Mes"] = df["Contrato"].str[3].map(Config.MAP_MESES)

    # Año = posiciones 5-6 → 20xx
    df["Año"] = df["Contrato"].str[4:6].astype(int).apply(lambda x: 2000 + x)

    return df


def cargar_datos_negociacion():
    """
    Carga y organiza los datos de negociación electrónica desde un archivo Excel.

    Retorna
    -------
    pd.DataFrame
        DataFrame con los datos organizados de negociación electrónica.
    """
    # Cargar datos
    df_neg = pd.read_excel(os.path.join(Config.DATA_PATH, "Neg_Electronica.xls"), sheet_name="Mejores Puntas")

    # Organizar datos
    df_neg = df_neg.rename(
        columns={
            "MEJOR PRECIO DE COMPRA (BID)": "BID",
            "CANTIDAD DE CONTRATOS CON EL MEJOR PRECIO DE COMPRA": "CABID",
            "MEJOR PRECIO DE VENTA (OFFER)": "OFFER",
            "CANTIDAD DE CONTRATOS CON EL MEJOR PRECIO DE VENTA": "CAOFFER"
        }
    )

    df_neg = transformar_neg(df_neg)

    # Renombrar columnas
    df_neg = df_neg.rename(
        columns={
            "Contrato": "Nemotecnico"
        }
    )
    df_neg = df_neg[~df_neg["Nemotecnico"].str.startswith("ELB")]

    # Columnas nuevas
    nuevas = ["Tipo", "Mes", "Año"]

    # Lista actual de columnas
    cols = list(df_neg.columns)

    # Posición donde está la columna "Convocado"
    idx = cols.index("Nemotecnico")

    # Construir nuevo orden:
    #   - todas las columnas antes de Convocado
    #   - Convocado
    #   - nuevas columnas
    #   - el resto
    nuevo_orden = (
        cols[:idx+1] +
        nuevas +
        [c for c in cols if c not in nuevas and c not in cols[:idx+1]]
    )

    # Reordenar dataframe
    df_neg = df_neg[nuevo_orden]

    return df_neg


## Negociación mixta
def cargar_datos_mixta():
    """
    Carga los datos de negociación mixta desde un archivo Excel.
    """
    neg_mixta = pd.read_excel(os.path.join(Config.DATA_PATH, "Neg_Mixta.xlsx"), sheet_name="Órdenes_Mixta")

    return neg_mixta


## Últimas
def cargar_ultimas():
    """
    Carga y organiza los datos de las últimas negociaciones y el resumen desde un archivo Excel.
    
    Retorna
    -------
    tuple of pd.DataFrame
        DataFrames con los datos organizados de las últimas negociaciones y el resumen.
    """
    # Cargar datos
    ultimas = pd.read_excel(os.path.join(Config.DATA_PATH, "Ultimas.xlsx"), sheet_name="Negociaciones", usecols="A:J")
    resumen = pd.read_excel(os.path.join(Config.DATA_PATH, "Ultimas.xlsx"), sheet_name="Resumen", skiprows=1, usecols="B:D")

    # Organizar ultimas
    ultimas = ultimas.drop(columns=["Mes contrato"])
    ultimas= ultimas.rename(
        columns={
            "Instrumento": "Tipo"
        }
    )
    ultimas= ultimas.rename(
        columns={
            "Carga total kWh": "Carga Total",
            "Sesión de negociación": "Sesión",
            "Contrato": "Nemotecnico"
        }
    )

    ## Mes → se toma la 4ta letra del nemotécnico
    ultimas["Mes"] = ultimas["Nemotecnico"].str[3].map(Config.MAP_MESES)

    ## Año → se toma la posición 5 y 6, y se convierte a 20XX
    ultimas["Año"] = "20" + ultimas["Nemotecnico"].str[4:6]
    ultimas["Año"] = ultimas["Año"].astype(int)

    # Organizar resumen

    ## Eliminar total de tabla
    resumen = resumen[~resumen["Mes"].astype(str).str.contains("Total", case=False, na=False)]
    resumen["Mes"] = pd.to_datetime(resumen["Mes"], errors="coerce")

    ## Asegurar que Mes es fecha
    resumen["Mes"] = pd.to_datetime(resumen["Mes"], errors="coerce")

    ## Crear columna Año
    resumen["Año"] = resumen["Mes"].dt.year

    ## Crear columna con el nombre del mes
    resumen["MesNombre"] = resumen["Mes"].dt.month_name(locale="es_ES")

    ## Asegurar que Mes es fecha
    resumen["Mes"] = pd.to_datetime(resumen["Mes"], errors="coerce")

    ## Crear columna Año
    resumen["Año"] = resumen["Mes"].dt.year

    ## Crear columna con el nombre del mes
    resumen["MesNombre"] = resumen["Mes"].dt.month_name(locale="es_ES")

    resumen= resumen.rename(
        columns={
            "kWh - mes": "kWh",
            "GWh - mes": "GWh",
            "MesNombre": "Mes"
        }
    )
    
    return ultimas, resumen

## Cargar datos
def get_data_futuros():
    """
    Carga todos los datos relacionados con futuros desde archivos Excel.

    Retorna
    -------
    pd.DataFrame
        DataFrame con los datos cargados y organizados.
    """
    print("Cargando datos futuros...")
    # Precios cierre futuros
    list_file_precios_fut = [x for x in os.listdir(Config.DATA_PATH) if x.startswith("Cierre") and x.endswith(".xlsx")]
    list_df = []
    for file in list_file_precios_fut:
        df = cargar_datos_futuros(file)
        list_df.append(df)

    df_cierre_futuros = pd.concat(list_df, ignore_index=True)

    # Precios futuros formato wide
    ## 1. Calcular meses al vencimiento (Buckets)
    df_cierre_futuros['Meses_al_Vencimiento'] = (
        (df_cierre_futuros['FechaVencimientoContrato'].dt.year - df_cierre_futuros['Fecha'].dt.year) * 12 + 
        (df_cierre_futuros['FechaVencimientoContrato'].dt.month - df_cierre_futuros['Fecha'].dt.month)
    )
    
    df_cierre_futuros["Tipo_Contrato"] = np.where(
        df_cierre_futuros['Meses_al_Vencimiento'] <= 9,
        df_cierre_futuros["Tipo"] + "_Vencimiento_0" + df_cierre_futuros["Meses_al_Vencimiento"].astype(str) + "Meses",
        df_cierre_futuros["Tipo"] + "_Vencimiento_" + df_cierre_futuros["Meses_al_Vencimiento"].astype(str) + "Meses"
    )
    df_cierre_futuros = df_cierre_futuros.drop(columns=["Meses_al_Vencimiento"])

    ## 2. Pivotear a formato wide
    df_cierre_futuros_wide = df_cierre_futuros.pivot_table(
        index="Fecha", 
        columns="Tipo_Contrato", 
        values="Precio"
    ).reset_index()

    # Convocatorias
    #conv, refe = cargar_datos_convocatorias()

    # Negociación electrónica
    #df_neg = cargar_datos_negociacion()

    # Negociación mixta
    #neg_mixta = cargar_datos_mixta()

    # Ultimas
    #ultimas, resumen = cargar_ultimas()

    print("✅ Datos de futuros cargados correctamente")
    return df_cierre_futuros, df_cierre_futuros_wide