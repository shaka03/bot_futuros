import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from config import Config

class Visualizer:
    def __init__(self, save_dir=Config.IMG_DIR):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sns.set_theme(style="whitegrid")

    def plot_results(self, history, contract_name, agent_idx):
        # 1. PnL
        plt.figure(figsize=(10, 5))
        plt.plot(history['rewards'], label='Recompensa Acumulada')
        plt.title(f"Performance: {contract_name}")
        plt.legend()
        plt.savefig(f"{self.save_dir}/pnl_{contract_name}.png")
        plt.close()

        # 2. Hedge Ratios (Curva)
        plt.figure(figsize=(10, 6))
        actions = np.array(history['actions']) # (Steps, Agents, Curve)
        # Extraer solo este agente
        my_actions = actions[:, agent_idx, :]
        
        for i in range(Config.CURVE_SIZE):
            plt.plot(my_actions[:, i], label=f'Posición M{i+1}', alpha=0.7)
            
        plt.title(f"Evolución de Posiciones (Curva): {contract_name}")
        plt.ylabel("Posición (Target Ratio)")
        plt.xlabel("Días")
        plt.legend()
        plt.savefig(f"{self.save_dir}/positions_{contract_name}.png")
        plt.close()
    
    def plot_advanced_metrics(self, history):
        """
        Genera gráficos financieros avanzados: Equity, Drawdown y Heatmap.
        history['total_equity']: Lista con la suma de self.cash de todos los agentes.
        history['positions']: Array de numpy (T, N_Agents, Curve_Size).
        """
        # --- 1. Curva de Equidad (Total Portfolio Value) ---
        plt.figure(figsize=(10, 6))
        equity = np.array(history['total_equity'])
        plt.plot(equity, label='Valor del Portafolio (Equity)', color='#2ca02c', linewidth=2)
        # Línea de referencia del capital inicial
        plt.axhline(y=Config.INITIAL_CAPITAL, color='red', linestyle='--', label='Capital Inicial', alpha=0.7)
        plt.title('Crecimiento del Capital Total')
        plt.ylabel('Valor (COP)')
        plt.xlabel('Días de Trading')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.save_dir}/equity_curve.png")
        plt.close()

        # --- 2. Drawdown ---
        # Calcular Drawdown: (Valor Actual - Pico Máximo Histórico) / Pico Máximo
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        
        plt.figure(figsize=(10, 4))
        plt.fill_between(range(len(drawdown)), drawdown, 0, color='#d62728', alpha=0.3)
        plt.plot(drawdown, color='#d62728', linewidth=1)
        plt.title('Drawdown Histórico (%)')
        plt.ylabel('Caída desde Máximo')
        plt.xlabel('Días')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.save_dir}/drawdown.png")
        plt.close()

        # --- 3. Heatmap de Posiciones ---
        # Promediamos la curva (M1, M2, M3) para ver la exposición neta por contrato
        # positions shape: (Time, Agents, Curve) -> (Time, Agents)
        positions = np.array(history['actions']) # O 'positions' si guardas el estado real
        net_exposure = np.mean(positions, axis=2) 
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(net_exposure.T, cmap="RdBu", center=0, cbar_kws={'label': 'Posición Neta (Short <-> Long)'},
                    yticklabels=Config.CONTRACT_TYPES)
        plt.title('Mapa de Calor de Exposición por Contrato')
        plt.xlabel('Días de Trading')
        plt.ylabel('Tipo de Contrato')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/heatmap_positions.png")
        plt.close()