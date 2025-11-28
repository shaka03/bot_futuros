import pandas as pd
import numpy as np
import os
from config import Config

class DataProcessor:
    """
    Clase para cargar y procesar datos de energía y futuros.
    """
    def __init__(self):
        """
        Inicializa las rutas de los archivos de datos.
        """
        self.energy_path = Config.ENERGY_FILE
        self.futures_path = Config.FUTURES_FILE

    def load_and_process(self):
        """
        Carga y procesa los datos de energía y futuros para construir la curva de precios.
        
        Returns:
            pd.DataFrame: DataFrame con la curva de precios procesada.
        """
        print("Cargando y procesando datos...")
        
        # 1. Cargar Spot
        df_spot = pd.read_csv(self.energy_path)
        # Filtrar PB_Nal y promediar por día
        df_spot = df_spot[df_spot["CodigoVariable"] == "PB_Nal"].copy()
        df_spot["Fecha"] = pd.to_datetime(df_spot["FechaHora"]).dt.date
        spot_daily = df_spot.groupby("Fecha")["Valor"].mean().reset_index()
        spot_daily.columns = ["Fecha", "SpotPrice"]
        spot_daily.set_index("Fecha", inplace=True)

        # 2. Cargar Futuros
        df_fut = pd.read_csv(self.futures_path)
        df_fut["Fecha"] = pd.to_datetime(df_fut["Fecha"]).dt.date
        
        # Mapa de meses español a número para calcular vencimientos
        meses = {"Enero":1, "Febrero":2, "Marzo":3, "Abril":4, "Mayo":5, "Junio":6,
                 "Julio":7, "Agosto":8, "Septiembre":9, "Octubre":10, "Noviembre":11, "Diciembre":12}
        df_fut["MesNum"] = df_fut["Mes"].map(meses)
        
        # Crear columna de fecha de vencimiento (Aproximación fin de mes)
        df_fut["ExpirationDate"] = pd.to_datetime(dict(year=df_fut.Año, month=df_fut.MesNum, day=1)) + pd.offsets.MonthEnd(1)
        df_fut["ExpirationDate"] = df_fut["ExpirationDate"].dt.date

        # 3. Construir la Curva (Pivot Table compleja)
        # Queremos para cada día y cada tipo de contrato: M1, M2, M3
        unique_dates = sorted(list(set(spot_daily.index) & set(df_fut["Fecha"].unique())))
        
        processed_data = {}
        
        for current_date in unique_dates:
            # Filtramos futuros disponibles en esa fecha
            day_futures = df_fut[df_fut["Fecha"] == current_date]
            
            # Datos base del día
            row_data = {"SpotPrice": spot_daily.loc[current_date, "SpotPrice"]}
            
            for c_type in Config.CONTRACT_TYPES:
                # Contratos de este tipo vigentes (Vencimiento > Fecha Actual)
                contracts = day_futures[
                    (day_futures["Tipo"] == c_type) & 
                    (day_futures["ExpirationDate"] > current_date)
                ].sort_values("ExpirationDate")
                
                # Rellenar curva M1, M2...
                for i in range(Config.CURVE_SIZE):
                    key_p = f"{c_type}_M{i+1}_Price"
                    key_d = f"{c_type}_M{i+1}_DTM" # Días al vencimiento
                    
                    if i < len(contracts):
                        ct = contracts.iloc[i]
                        row_data[key_p] = ct["Precio"]
                        row_data[key_d] = (ct["ExpirationDate"] - current_date).days
                    else:
                        # No hay contrato disponible (ej. fin de datos)
                        row_data[key_p] = 0.0
                        row_data[key_d] = 0.0
            
            processed_data[current_date] = row_data
            
        df_final = pd.DataFrame.from_dict(processed_data, orient="index")
        df_final.index = pd.to_datetime(df_final.index)
        df_final.sort_index(inplace=True)
        
        # Rellenar huecos (forward fill)
        df_final.fillna(method="ffill", inplace=True)
        df_final.dropna(inplace=True) # Eliminar filas iniciales vacías
        
        return df_final