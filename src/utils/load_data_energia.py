#%% Librerías
from pydataxm.pydatasimem import CatalogSIMEM, ReadSIMEM
import datetime as dt
import pandas as pd

"""
Identificadores de datasets SIMEM de energía:
- Demanda eléctrica: "d55202"
- Precios de mercado: "EC6945"
- Precios ponderados: "96D56E"
- Aportes hídricos: "BA1C55"
- Niveles de embalse: "BD26DC"
- Disponibilidad real: "9E77E5"
- Generación real: "E17D25"
"""

class DatasetSIMEM:
    ID_DICT = {
        "DEMANDA": "d55202",
        "PRECIOS": "EC6945",
        "PRECIOS_PONDERADOS": "96D56E",
        "APORTES_HIDRICOS": "BA1C55",
        "NIVELES_EMBALSE": "BD26DC",
        "GENERACION_REAL": "E17D25",
        "DISPONIBILIDAD_REAL": "9E77E5"
    }

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
        data_path: str = "data/raw"
) -> dict:
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
    print("Cargando datos de XM...")
    for nombre, id_dataset in DatasetSIMEM.ID_DICT.items():
        print(f"Cargando {nombre}...")
        df = get_data(id_dataset, fecha_inicio_str, fecha_fin_str)
        df.to_csv(f"{data_path}/datos_crudos_{nombre}.csv", index=False)
    
    print("✅ Datos de energía cargados correctamente")

    return None