import numpy as np
from config import Config
from data_processor import DataProcessor
from environment import EnergyTradingEnv
from maddpg import MADDPG
from visualization import Visualizer

def main():
    # 1. Datos
    dp = DataProcessor()
    df = dp.load_and_process()
    print(f"Datos procesados: {df.shape[0]} días de trading.")
    
    # Split Train/Test
    split = int(len(df) * Config.TRAIN_SPLIT_PCT)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    
    # 2. Entorno
    env_train = EnergyTradingEnv(train_df, train_mode=True)
    
    # Dimensiones
    n_agents = len(Config.CONTRACT_TYPES)
    obs_dims = [env_train.observation_space[i].shape[0] for i in range(n_agents)]
    act_dims = [env_train.action_space[i].shape[0] for i in range(n_agents)]
    
    # 3. Agente MADDPG
    maddpg = MADDPG(n_agents, obs_dims, act_dims)
    
    # 4. Entrenamiento
    print("Iniciando entrenamiento...")
    for episode in range(Config.EPISODES):
        obs = env_train.reset()
        episode_reward = 0
        
        while True:
            # Seleccionar acción (Exploración con ruido)
            actions = []
            for i in range(n_agents):
                # Flatten obs para el agente
                act = maddpg.agents[i].select_action(obs[i], noise=0.1)
                actions.append(act)
            
            actions = np.array(actions)
            
            # Paso
            next_obs, rewards, done, _ = env_train.step(actions)
            
            # Guardar en buffer
            maddpg.memory.add(obs, actions, rewards, next_obs, done)
            
            # Actualizar redes
            maddpg.update(Config.BATCH_SIZE)
            
            obs = next_obs
            episode_reward += np.sum(rewards)
            
            if done:
                print(f"Episodio {episode+1}: Recompensa Total {episode_reward:.2f}")
                break
                
    # 5. Testing
    print("Iniciando Prueba...")
    env_test = EnergyTradingEnv(test_df, train_mode=False)
    viz = Visualizer()
    
    obs = env_test.reset()
    history = {"rewards": [], "actions": [], "total_equity": []}
    
    while True:
        actions = []
        for i in range(n_agents):
            act = maddpg.agents[i].select_action(obs[i], noise=0.0) # Sin ruido
            actions.append(act)
        actions = np.array(actions)
        
        next_obs, rewards, done, _ = env_test.step(actions)
        
        # Guardar métricas
        total_cash = np.sum(env_test.cash) # Suma del dinero de todos los agentes
        history["total_equity"].append(total_cash)
        history["actions"].append(actions)
        obs = next_obs
        
        if done:
            break
            
    # Graficar
    for i, name in enumerate(Config.CONTRACT_TYPES):
        viz.plot_results(history, name, i)
        viz.plot_advanced_metrics(history)

if __name__ == "__main__":
    main()