import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#%% Configuración global para la simulación
class Config:
    SIMULATIONS = 1000        # Número de caminos simulados para Monte Carlo
    DAYS_TO_FORECAST = 365    # Número de días a pronosticar (1 año)
    ALIGNMENT_LAG = 364      # 52 semanas exactas para alinear días de la semana (Lunes con Lunes)
    TREND_WINDOW_DAYS = 120   # Días recientes para calcular la tendencia actual (4 meses aprox.)
    RANDOM_SEED = 42         # Semilla para reproducibilidad

#%% Funciones par simular demanda
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
        
        np.random.seed(Config.RANDOM_SEED)  # Para reproducibilidad
        
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