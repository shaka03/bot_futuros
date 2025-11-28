import pandas as pd
import numpy as np
from config import Config

class DataProcessor:
    def __init__(self):
        self.energy_path = Config.ENERGY_FILE
        self.futures_path = Config.FUTURES_FILE

    def load_and_process(self):
        # 1. Cargar y Procesar Precio Spot (Energía)
        df_spot = pd.read_csv(self.energy_path)
        # Filtrar precio de bolsa nacional y agrupar por día (promedio aritmético)
        df_spot = df_spot[df_spot[Config.COD_VARIABLE_COL].isin(Config.COD_VARIABLE)].copy()
        df_spot[Config.FECHA_SPOT_COL] = pd.to_datetime(df_spot[Config.FECHA_SPOT_COL])
        df_spot["Fecha"] = df_spot[Config.FECHA_SPOT_COL].dt.date
        # Promedio diario del precio spot
        spot_daily = df_spot.groupby("Fecha")["Valor"].mean().reset_index()
        spot_daily.columns = ["Fecha", "SpotPrice"]
        spot_daily["Fecha"] = pd.to_datetime(spot_daily["Fecha"])
        spot_daily.set_index("Fecha", inplace=True)

        # 2. Cargar y Procesar Futuros
        df_fut = pd.read_csv(self.futures_path)
        df_fut[Config.FECHA_FUT_COL] = pd.to_datetime(df_fut[Config.FECHA_FUT_COL])
        df_fut[Config.PRECIO_COL] = pd.to_numeric(df_fut[Config.PRECIO_COL], errors="coerce")
        df_fut.dropna(subset=[Config.PRECIO_COL], inplace=True)

        # Crear una columna de fecha de vencimiento aproximada
        # Mapeo de meses a números
        meses = {"Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
                 "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12}
        df_fut["MesNum"] = df_fut["Mes"].map(meses)
        
        # Asumimos vencimiento el último día del mes para simplificar la construcción de la curva
        df_fut["Vencimiento"] = pd.to_datetime(dict(year=df_fut["Año"], month=df_fut["MesNum"], day=1)) + pd.offsets.MonthEnd(1)

        # 3. Construir Curva Continua (M1, M2) por Tipo de Contrato
        # Pivotar para tener: Index=Fecha, Columns=(Tipo, Vencimiento), Values=Precio
        pivot_fut = df_fut.pivot_table(index="Fecha", columns=["Tipo", "Vencimiento"], values="Precio")
        
        # Alinear fechas con Spot
        full_data = spot_daily.join(pivot_fut, how="inner")
        full_data.sort_index(inplace=True)
        
        return full_data

    def get_state_for_date(self, full_data, current_date, contract_type):
        """
        Extrae el estado para un contrato específico en una fecha dada sin Look-ahead bias.
        Retorna: Spot, Precio_M1, Precio_M2, Dias_Vencimiento_M1
        """
        if current_date not in full_data.index:
            return None
            
        row = full_data.loc[current_date]
        spot = row["SpotPrice"]
        
        # Filtrar columnas del contrato específico
        # Las columnas son MultiIndex (Tipo, Vencimiento)
        cols = [c for c in full_data.columns if isinstance(c, tuple) and c[0] == contract_type]
        
        # Encontrar contratos vigentes (Vencimiento > Fecha Actual)
        valid_contracts = []
        for _, expiration in cols:
            if expiration > current_date:
                price = row[(contract_type, expiration)]
                if not np.isnan(price):
                    valid_contracts.append((expiration, price))
        
        # Ordenar por vencimiento más cercano
        valid_contracts.sort(key=lambda x: x[0])
        
        if len(valid_contracts) == 0:
            return np.array([spot, 0, 0, 0]) # No hay futuros disponibles
            
        m1_exp, m1_price = valid_contracts[0]
        m2_price = valid_contracts[1][1] if len(valid_contracts) > 1 else m1_price
        
        days_to_maturity = (m1_exp - current_date).days
        
        # Normalización simple (se puede mejorar con StandardScaler ajustado solo en train)
        return np.array([spot, m1_price, m2_price, days_to_maturity])