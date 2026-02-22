#%% Librerías

# Librerías úitiles
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


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
        "disponibilidad": "datos_DISPONIBILIDAD_REAL.csv"
    }

    SIMULATIONS = 1000       
    DAYS_TO_FORECAST = 365   
    ALIGNMENT_LAG = 364      # 52 semanas exactas para alinear días de la semana (Lunes con Lunes)
    TREND_WINDOW_DAYS = 120

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

# Funciones par simular demanda
def get_trend_slope(series, days):
    """
    Calcula la pendiente de crecimiento diario (kWh/día) 
    basado en los últimos 'days' días usando regresión lineal.
    """
    y = series.iloc[-days:].values
    X = np.arange(len(y)).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # La pendiente (coef_) nos dice cuánto crece la demanda por día en promedio
    return model.coef_[0]

def run_simulation(
        df: pd.DataFrame
) -> pd.DataFrame:
    df = df.set_index('Fecha')
    df = df.sort_index()
    target_cols = df.columns
    list_df = []
    
    for col in target_cols:
        print(f"  Simulando demanda para: {col}")
        
        series = df[col]
        
        # 1. ANÁLISIS DE TENDENCIAS
        # -------------------------
        # A. Tendencia Reciente (Target): ¿Qué tan rápido estamos creciendo HOY?
        # Usamos 120 días para capturar la tendencia positiva sólida reciente
        target_slope = get_trend_slope(series, days=Config.TREND_WINDOW_DAYS)
        
        # B. Tendencia Histórica (Base): ¿Qué tan rápido crecía el AÑO PASADO?
        # Calculamos la pendiente del mismo periodo pero hace un año
        hist_slope = get_trend_slope(series.iloc[:-Config.DAYS_TO_FORECAST], days=Config.ALIGNMENT_LAG)
        
        # C. Ajuste (Drift): La diferencia.
        # Si hoy crecemos a 500 kWh/día y el año pasado a 100, 
        # debemos sumar 400 kWh extra cada día a la forma del año pasado.
        drift_adjustment = target_slope - hist_slope
        
        # REGLA DE SEGURIDAD PARA 7-17 Y OTRAS:
        # Si el usuario confirma que la tendencia es positiva, forzamos que el ajuste no sea negativo
        # (evita que un año pasado "malo" nos tumbe el pronóstico)
        if drift_adjustment < 0 and target_slope > 0:
             drift_adjustment = target_slope # Ignoramos la historia negativa, imponemos el crecimiento actual
        
        # 2. CONSTRUCCIÓN DE LA SIMULACIÓN
        # --------------------------------
        # Método: "Step-Forward" (Paso adelante)
        # Pronóstico[t] = Pronóstico[t-1] + (Cambio_Hace_Un_Año) + (Ajuste_Tendencia)
        
        # Extraemos los cambios diarios del año pasado (Deltas)
        # Esto captura la estacionalidad (Lunes sube, Domingo baja, Enero sube, etc.)
        last_year_data = series.iloc[-Config.ALIGNMENT_LAG-1:].values
        
        # Calculamos día a día cuánto cambió la demanda hace 364 días
        historical_deltas = []
        for i in range(Config.DAYS_TO_FORECAST):
            # Índice correspondiente al día simétrico del año pasado
            # +1 porque diff necesita el previo
            idx = i + 1 
            # Protección por si el forecast es más largo que la historia (loop)
            if idx >= len(last_year_data):
                idx = idx % Config.ALIGNMENT_LAG + 1
            
            # Delta = Valor[i] - Valor[i-1]
            delta = last_year_data[idx] - last_year_data[idx-1]
            historical_deltas.append(delta)
            
        historical_deltas = np.array(historical_deltas)
        
        # 3. MONTE CARLO
        # --------------
        simulated_paths = np.zeros((Config.DAYS_TO_FORECAST, Config.SIMULATIONS))
        
        # Volatilidad para el ruido aleatorio
        daily_vol = series.pct_change().dropna().std()
        current_val = series.iloc[-1]
        
        np.random.seed(42)
        
        # Matriz de Ruido: (Días x Simulaciones)
        # El ruido es proporcional al valor actual (aprox)
        noise_matrix = np.random.normal(loc=0, scale=daily_vol * current_val, size=(Config.DAYS_TO_FORECAST, Config.SIMULATIONS))
        
        # Matriz de Tendencia: 
        # Agregamos un poco de incertidumbre a la pendiente también (+/- 10%)
        slope_uncertainty = np.random.normal(loc=0, scale=abs(drift_adjustment)*0.1, size=(1, Config.SIMULATIONS))
        total_drift_matrix = drift_adjustment + slope_uncertainty
        
        # Matriz Base (Deltas históricos repetidos)
        base_deltas_matrix = np.tile(historical_deltas.reshape(-1, 1), (1, Config.SIMULATIONS))
        
        # CAMBIO TOTAL DIARIO = (Patrón Año Pasado) + (Tendencia Extra) + (Ruido)
        all_daily_changes = base_deltas_matrix + total_drift_matrix + noise_matrix
        
        # ACUMULAR CAMBIOS DESDE EL ÚLTIMO VALOR REAL (ENLACE PERFECTO)
        # Empezamos con el último valor real y vamos sumando los cambios acumulados
        cumulative_changes = np.cumsum(all_daily_changes, axis=0)
        simulated_paths = current_val + cumulative_changes
        
        # 4. RESULTADOS
        # ------------------------
        future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=Config.DAYS_TO_FORECAST)
        
        p50 = np.percentile(simulated_paths, 50, axis=1)
        
        results_df = pd.DataFrame({
            "Fecha": future_dates,
            f"{col}": p50
        })

        list_df.append(results_df)
    
    # Unimos todas las simulaciones en un solo DataFrame
    df_simulaciones = list_df[0]
    for df_temp in list_df[1:]:
        df_simulaciones = pd.merge(df_simulaciones, df_temp, on="Fecha", how="outer")
    
    # Concatenamos con los datos reales para tener un dataset completo
    df_completo = pd.concat([df.reset_index(), df_simulaciones], ignore_index=True)
    df_completo.sort_values("Fecha", inplace=True)

    return df_completo

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
                df.to_csv(os.path.join(Config.GOLD_DATA_PATH, "datos_FUTUROS.csv"), index=False)

                # Guardar fechas transacciones
                df_fechas = df[df["Fecha"] >= fecha_inicio_transacciones_str][["Fecha"]]
                df_fechas = df_fechas.drop_duplicates().sort_values("Fecha").reset_index(drop=True)
                df_fechas.to_csv(os.path.join(Config.GOLD_DATA_PATH, "fechas_transacciones.csv"), index=False)

                # Completar fechas con el ultimo precio conocido (forward fill)
                fechas_completas = pd.DataFrame({"FechaCompleta": pd.date_range(start=fecha_inicio_str, end=fecha_fin_str)})
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
                rolling_var = df_contrato["Retorno_Futuros"].rolling(window="30D", min_periods=30).var()
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
        df_futuros_final.to_csv(os.path.join(Config.GOLD_DATA_PATH, "precios_FUTUROS.csv"), index=False)

    # Cargar datos noticias
    
    print("Datasets finales guardado en GOLD.")
    return None


#%% Ejecutar carga de datos
if __name__ == "__main__":
    fecha_inicio_str = "2022-01-01"
    fecha_inicio_transacciones_str = "2022-02-01"
    fecha_fin_str = "2026-01-31"
    process_data(fecha_inicio_str, fecha_fin_str, fecha_inicio_transacciones_str)