import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np
import os
from config import Config

class Visualizer:
    def __init__(self, save_dir=Config.SAVE_DIR):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Configuración de estilo
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 12

    def plot_cumulative_pnl(self, df_results):
        """
        Grafica la curva de PnL acumulado del Agente vs Benchmark.
        
        Args:
            df_results (pd.DataFrame): Debe contener 'Fecha', 'Agent_PnL', 'Benchmark_PnL'
        """
        plt.figure()
        
        # Calcular acumulados
        df_results['Agent_Cum'] = df_results['Agent_PnL'].cumsum()
        df_results['Bench_Cum'] = df_results['Benchmark_PnL'].cumsum()
        
        plt.plot(df_results['Fecha'], df_results['Agent_Cum'], label='Dual Q Agent', linewidth=2, color='blue')
        plt.plot(df_results['Fecha'], df_results['Bench_Cum'], label='Buy & Hold (Benchmark)', linewidth=2, color='gray', linestyle='--')
        
        plt.title("Evolución de Ganancia/Pérdida Acumulada (PnL)")
        plt.xlabel("Fecha")
        plt.ylabel("PnL Acumulado (COP)")
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        path = os.path.join(self.save_dir, 'pnl_comparison.png')
        plt.savefig(path)
        print(f"Gráfico guardado en: {path}")
        plt.close()

    def plot_hedge_ratio_vs_spot(self, df_results, contract_name):
        """
        Gráfico de doble eje: Precio Spot vs Hedge Ratio para un contrato específico.
        Permite ver si el agente 'persigue' el precio o hace reversión a la media.
        """
        fig, ax1 = plt.subplots()
        
        # Eje 1: Precio Spot
        color = 'tab:red'
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Precio Spot (COP/kWh)', color=color)
        ax1.plot(df_results['Fecha'], df_results['Spot_Price'], color=color, alpha=0.6, label='Precio Spot')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Eje 2: Hedge Ratio
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        col_name = f'Hedge_Ratio_{contract_name}'
        
        if col_name not in df_results.columns:
            print(f"Advertencia: No se encontró columna {col_name}")
            return

        ax2.set_ylabel(f'Hedge Ratio ({contract_name})', color=color)  
        # Usamos step para mostrar cambios discretos en la decisión
        ax2.step(df_results['Fecha'], df_results[col_name], color=color, where='post', linewidth=1.5, label='Agente')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-1.1, 1.1) # Límites fijos para ver saturación
        
        # Línea cero para referencia
        ax2.axhline(0, color='black', linewidth=0.5, linestyle=':')

        plt.title(f"Comportamiento del Agente: {contract_name} vs Mercado Spot")
        fig.tight_layout()
        
        path = os.path.join(self.save_dir, f'behavior_{contract_name}.png')
        plt.savefig(path)
        print(f"Gráfico guardado en: {path}")
        plt.close()

    def plot_scatter_correlation(self, df_results, contract_name):
        """
        Scatter plot para ver la correlación directa entre Nivel de Precios y Posición.
        """
        plt.figure(figsize=(8, 8))
        col_name = f'Hedge_Ratio_{contract_name}'
        
        sns.scatterplot(data=df_results, x='Spot_Price', y=col_name, alpha=0.5)
        plt.title(f"Correlación: Precio Spot vs Posición ({contract_name})")
        plt.xlabel("Precio Spot")
        plt.ylabel("Hedge Ratio")
        plt.ylim(-1.1, 1.1)
        
        path = os.path.join(self.save_dir, f'scatter_{contract_name}.png')
        plt.savefig(path)
        plt.close()