import numpy as np
import pandas as pd
from config import Config
from data_processor import DataProcessor
from environment import EnergyHedgingEnv
from maddpg import MADDPG
from visualization import Visualizer

def main():
    # 1. Preparar Datos
    print("Procesando datos...")
    dp = DataProcessor()
    full_data = dp.load_and_process()
    
    # Split Temporal (Train / Test)
    all_dates = full_data.index.unique().sort_values()
    split_idx = int(len(all_dates) * (1 - Config.TEST_PCT))
    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]
    
    print(f"Datos Entrenamiento: {len(train_dates)} días")
    print(f"Datos Prueba: {len(test_dates)} días")

    # 2. Inicializar Entorno y Agentes
    env = EnergyHedgingEnv(full_data, train_dates, train_mode=True)
    
    n_agents = len(Config.CONTRACT_TYPES)
    state_dims = [env.observation_space[i].shape[0] for i in range(n_agents)]
    action_dims = [env.action_space[i].shape[0] for i in range(n_agents)]
    
    maddpg = MADDPG(n_agents, state_dims, action_dims)
    
    # 3. Loop de Entrenamiento
    print("Iniciando entrenamiento MADDPG Dual Q...")
    for episode in range(Config.EPISODES):
        obs = env.reset()
        total_reward = 0
        
        while True:
            # Seleccionar acciones (Exploración con ruido)
            actions = []
            for i in range(n_agents):
                act = maddpg.agents[i].select_action(obs[i], noise=0.1)
                actions.append(act)
            
            # Paso del entorno
            next_obs, rewards, done, info = env.step(actions)
            
            # Guardar en memoria y entrenar
            maddpg.add_to_memory(obs, actions, rewards, next_obs, done)
            maddpg.update()
            
            obs = next_obs
            total_reward += sum(rewards)
            
            if done:
                break
        
        if episode % 5 == 0:
            print(f"Episodio {episode}: Recompensa Total: {total_reward:.2f}")

    # 4. Loop de Prueba (Backtest con recolección de métricas)
    print("\nIniciando Backtest detallado...")
    env_test = EnergyHedgingEnv(full_data, test_dates, train_mode=False)
    obs = env_test.reset()
    
    # Estructura para guardar historia
    history = {
        'Fecha': [],
        'Agent_PnL': [],
        'Benchmark_PnL': [],
        'Spot_Price': []
    }
    # Columnas dinámicas para cada agente
    for c_name in Config.CONTRACT_TYPES:
        history[f'Hedge_Ratio_{c_name}'] = []
    
    # Variables para Benchmark "Buy & Hold" (Posición constante de 1.0)
    # Simulamos un agente tonto que siempre tiene +1 contrato
    bench_position = np.ones(n_agents) 
    
    step_idx = 0
    while True:
        current_date = test_dates[step_idx]
        
        # 1. Obtener Acciones del Agente
        actions = []
        agent_actions_record = {}
        for i in range(n_agents):
            act = maddpg.agents[i].select_action(obs[i], noise=0.0)
            actions.append(act)
            # Guardar para graficar (act[0] es el Hedge Ratio)
            agent_actions_record[f'Hedge_Ratio_{Config.CONTRACT_TYPES[i]}'] = act[0]
        
        # 2. Calcular PnL del Benchmark (Simulado manualmente)
        # Necesitamos ver el cambio de precio 'mañana' para calcular el PnL de hoy
        # Como env.step calcula el PnL basado en la transición, usamos una lógica similar
        # Nota: Esto es una aproximación, idealmente el Env retornaría precios crudos
        
        # Guardar estado antes del step
        spot_price_today = obs[0][0] # Asumiendo que Spot es index 0 en el estado
        
        # 3. Paso del Entorno
        next_obs, rewards, done, info = env_test.step(actions)
        
        # El environment ya calcula el PnL del agente en 'rewards' (o info['pnl'])
        # info['pnl'] es el cambio de valor del portafolio del agente
        agent_daily_pnl = info['pnl']
        
        # --- Cálculo Benchmark (Buy & Hold) ---
        # Aproximación: Si Buy & Hold mantiene 1.0, su PnL es simplemente (Precio_Mañana - Precio_Hoy)
        # Usamos el dato del environment o reconstruimos con precios
        # Para simplificar sin alterar Environment: Usamos el cambio de precio implícito
        # Si el agente tuviera posición 1.0, su PnL sería X. 
        # Podemos inferir el cambio de precio si sabemos la posición del agente anterior, pero es complejo.
        # MEJOR OPCIÓN: Comparar contra PnL cero (No Hedge) o asumir Benchmark aleatorio para el ejemplo.
        # Aquí calcularemos un Benchmark simple: El agente gana X.
        # Asumiremos que Buy&Hold gana proporcional al cambio del spot (proxy de futuros).
        try:
            spot_next = next_obs[0][0]
            bench_daily_pnl = (spot_next - spot_price_today) * 10 # 10 contratos fijos
        except:
            bench_daily_pnl = 0 # Último día
            
        # 4. Guardar Datos
        history['Fecha'].append(current_date)
        history['Agent_PnL'].append(agent_daily_pnl)
        history['Benchmark_PnL'].append(bench_daily_pnl)
        history['Spot_Price'].append(spot_price_today)
        
        for k, v in agent_actions_record.items():
            history[k].append(v)
            
        obs = next_obs
        step_idx += 1
        
        if done:
            break

    # Crear DataFrame
    df_results = pd.DataFrame(history)
    
    # Generar Visualizaciones
    print("Generando gráficos...")
    viz = Visualizer()
    
    # 1. PnL Comparativo
    viz.plot_cumulative_pnl(df_results)
    
    # 2. Comportamiento por Contrato
    for c_name in Config.CONTRACT_TYPES:
        viz.plot_hedge_ratio_vs_spot(df_results, c_name)
        viz.plot_scatter_correlation(df_results, c_name)
        
    print("Proceso finalizado. Revise la carpeta 'resultados_img'.")

if __name__ == "__main__":
    main()