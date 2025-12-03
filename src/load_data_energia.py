#%% Librerías
from pydataxm.pydatasimem import CatalogSIMEM, ReadSIMEM
import datetime as dt
import pandas as pd

"""
Identificadores de datasets SIMEM de energía:
- Demanda eléctrica: "14fabb"
- Precios de mercado: "EC6945"
- Aportes hídricos: "BA1C55"
- Generación real: "055A4D"
"""

#%% Funciones
def get_data(
        id_dataset: str,
        fecha_inicio_str: str = "2024-01-01",
        fecha_fin_str: str = dt.datetime.now().strftime("%Y-%m-%d"),

) -> pd.DataFrame:
    """
    Carga datos de SIMEM en un DataFrame de pandas.

    Parámetros
    ----------
    id_dataset : str
        Identificador del dataset en SIMEM.
    fecha_incio_str : str, opcional
        Fecha de inicio en formato "YYYY-MM-DD". Por defecto es "2024-01-01".
    fecha_fin_str : str
        Fecha de fin en formato "YYYY-MM-DD". Por defecto es la fecha actual.

    Retorna
    -------
    pd.DataFrame
        DataFrame con los datos cargados desde SIMEM.
    """
    fecha_inicio = dt.datetime.strptime(fecha_inicio_str, "%Y-%m-%d")
    fecha_fin = dt.datetime.strptime(fecha_fin_str, "%Y-%m-%d")
    list_df = []
    keep = True

    while keep:
        fecha_part = dt.datetime(fecha_inicio.year, 12, 31)
        fecha_part_str = dt.datetime.strftime(fecha_part, "%Y-%m-%d")
        print(f"{fecha_inicio_str} a {fecha_part_str}")
        if fecha_part >= fecha_fin:
            reader = ReadSIMEM(id_dataset, fecha_inicio_str, fecha_fin_str)
            keep = False
        else:
            reader = ReadSIMEM(id_dataset, fecha_inicio_str, fecha_part_str)

        if id_dataset == "EC6945":
            df = reader.main(filter=False)
            df = df[df["CodigoVariable"] == "PB_Nal"]
        
        df = reader.main(filter=False)
        list_df.append(df)
        del df

        fecha_inicio = dt.datetime(fecha_inicio.year + 1, 1, 1)
        fecha_inicio_str = dt.datetime.strftime(fecha_inicio, "%Y-%m-%d")

    df = pd.concat(list_df)
    return df


def load_data(
        fecha_inicio_str: str = "2024-01-01",
        fecha_fin_str: str = dt.datetime.now().strftime("%Y-%m-%d"),
) -> pd.DataFrame:
    """
    Carga datos de SIMEM y los guarda en un archivo CSV.

    Parámetros
    ----------
    fecha_incio_str : str, opcional
        Fecha de inicio en formato "YYYY-MM-DD". Por defecto es "2024-01-01".
    fecha_fin_str : str
        Fecha de fin en formato "YYYY-MM-DD". Por defecto es la fecha actual.

    Retorna
    -------
    pd.DataFrame
        DataFrame con los datos cargados desde SIMEM.
    """
    print("Cargando datos de energía...")
    # Precios de mercado
    ## Precos por hora
    id_dataset_precios = "EC6945"
    df_precios = get_data(id_dataset_precios, fecha_inicio_str, fecha_fin_str)

    # Demanda eléctrica
    ## Demanda por hora, regulada y no regulada
    #id_dataset_demanda = "14fabb"
    #df_demanda = get_data(id_dataset_demanda, fecha_inicio_str, fecha_fin_str)

    # Aportes hídricos
    ## Aportes por cuenca, por día
    #id_dataset_aportes = "BA1C55"
    #df_aportes = get_data(id_dataset_aportes, fecha_inicio_str, fecha_fin_str)

    # Generación
    ## Generación por hora, por planta y agente
    #id_dataset_generacion = "055A4D"
    #df_generacion = get_data(id_dataset_generacion, fecha_inicio_str, fecha_fin_str)
    
    print("✅ Datos de energía cargados correctamente")

    return df_precios#, df_demanda, df_aportes, df_generacion