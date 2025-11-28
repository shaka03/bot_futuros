import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import Config

class EnergyTradingEnv(gym.Env):
    def __init__(self, df, train_mode=True):
        super(EnergyTradingEnv, self).__init__()
        self.df = df
        self.dates = df.index
        self.train_mode = train_mode
        self.n_agents = len(Config.CONTRACT_TYPES)
        
        # Acciones: Target Hedge Ratio para cada contrato en la curva (M1, M2, M3)
        # Rango [-1, 1] (Vender Futuro -> Hedging normal, Comprar -> Especulación)
        self.action_space = [spaces.Box(low=-1, high=1, shape=(Config.CURVE_SIZE,), dtype=np.float32) 
                             for _ in range(self.n_agents)]
        
        # Estado:
        # 1. Spot Price (1)
        # 2. Futures Prices (Curve Size)
        # 3. DTM (Curve Size)
        # 4. Current Positions (Curve Size)
        # Total = 1 + 3*CURVE_SIZE
        self.obs_dim = 1 + 3 * Config.CURVE_SIZE
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32) 
                                  for _ in range(self.n_agents)]
        
        self.reset()

    def reset(self):
        self.current_idx = 0 # Empezamos en 0, pero obs usa lógica de T-1 si es necesario
        self.positions = np.zeros((self.n_agents, Config.CURVE_SIZE)) # Contratos poseídos
        initial_cash_per_agent = Config.INITIAL_CAPITAL / self.n_agents
        self.cash = np.full(self.n_agents, initial_cash_per_agent)
        self.portfolio_history = []
        return self._get_obs()

    def _get_obs(self):
        # Observación disponible a las 8 AM del día "current_idx"
        # Usamos los precios de cierre de "current_idx" como si fueran los de ayer 
        # (Asumiendo que df tiene filas T-1, T, T+1...)
        # NOTA: Para evitar leakage estricto, en el "step" avanzamos el índice ANTES de calcular PnL
        
        obs_list = []
        row = self.df.iloc[self.current_idx]
        
        for i, c_type in enumerate(Config.CONTRACT_TYPES):
            # Precios Futuros
            prices = [row[f"{c_type}_M{k+1}_Price"] for k in range(Config.CURVE_SIZE)]
            # Dias Vencimiento
            dtms = [row[f"{c_type}_M{k+1}_DTM"] for k in range(Config.CURVE_SIZE)]
            
            # Normalización simple para la red neuronal
            state = np.concatenate([
                [row["SpotPrice"] / 1000.0],
                np.array(prices) / 1000.0,
                np.array(dtms) / 365.0,
                self.positions[i] # Posiciones actuales
            ])
            obs_list.append(state)
            
        return np.array(obs_list)

    def step(self, actions):
        # actions: (n_agents, curve_size) -> Target Ratios
        
        # 1. Datos actuales (8 AM - Momento decisión)
        today_row = self.df.iloc[self.current_idx]
        
        rewards = []
        next_positions = np.zeros_like(self.positions)
        
        # --- LÓGICA DE TRANSACCIÓN (8 AM) ---
        transaction_costs = np.zeros(self.n_agents)
        
        for i, c_type in enumerate(Config.CONTRACT_TYPES):
            agent_action = actions[i] # Target Ratios
            current_pos = self.positions[i]
            
            # Precios de ejecución (asumimos precio de cierre de ayer como proxy de apertura o precio fijo)
            # Para mayor realismo, si tuviéramos datos OHLC, usaríamos Open.
            prices = np.array([today_row[f"{c_type}_M{k+1}_Price"] for k in range(Config.CURVE_SIZE)])
            
            # Cambio de posición
            # Asumimos que la acción es el TARGET position (no el delta)
            # Position size arbitraria base: 10 contratos por unidad de acción
            target_contracts = agent_action * 10 
            delta_contracts = target_contracts - current_pos
            
            # Costo de transacción
            cost = np.sum(np.abs(delta_contracts) * prices * Config.TRANSACTION_FEE)
            transaction_costs[i] = cost
            
            next_positions[i] = target_contracts

        # --- AVANCE DEL TIEMPO (Market Close) ---
        self.current_idx += 1
        done = self.current_idx >= (len(self.df) - 1)
        
        if done:
            # Fin del episodio
            return self._get_obs(), np.zeros(self.n_agents), True, {}

        # 2. Datos de Cierre (Día T al final)
        tomorrow_row = self.df.iloc[self.current_idx] 
        # (Nota: tomorrow_row es el cierre del día donde tomamos la decisión)

        for i, c_type in enumerate(Config.CONTRACT_TYPES):
            # Calcular Mark-to-Market PnL
            
            # Precios viejos (8 AM)
            prices_old = np.array([today_row[f"{c_type}_M{k+1}_Price"] for k in range(Config.CURVE_SIZE)])
            # Precios nuevos (Cierre)
            prices_new = np.array([tomorrow_row[f"{c_type}_M{k+1}_Price"] for k in range(Config.CURVE_SIZE)])
            dtms = np.array([tomorrow_row[f"{c_type}_M{k+1}_DTM"] for k in range(Config.CURVE_SIZE)])
            
            pos = next_positions[i]
            
            # PnL Diario por variacion de precio
            mtm_pnl = np.sum(pos * (prices_new - prices_old))
            
            # --- LIQUIDACIÓN / VENCIMIENTO ---
            # Si DTM <= 0, el contrato vence.
            # En la vida real, M1 vence y desaparece, M2 se vuelve M1.
            # Aquí simplificamos: Si M1 vence, liquidamos contra Spot.
            
            settlement_pnl = 0
            # Revisar curva:
            # Si la curva cambia (M1 expira), necesitamos ajustar "pos" para el siguiente estado
            # Pero como la acción es TARGET, el agente reajustará en el siguiente step.
            # Lo importante es el PnL financiero hoy.
            
            if dtms[0] <= 0: # M1 Venció
                # Diferencia entre Precio Futuro Final y Spot
                # Liquidación financiera: (Spot - Futuro) * Posición (para vendedor)
                # O simplemente: La posición converge al Spot.
                spot = tomorrow_row["SpotPrice"]
                # El PnL final es la convergencia
                settlement_pnl = pos[0] * (spot - prices_new[0]) 
                # (Nota: prices_new[0] debería estar muy cerca de spot si el mercado es eficiente)
            
            total_pnl = mtm_pnl + settlement_pnl - transaction_costs[i]

            # Actualizar Caja
            self.cash[i] += total_pnl 
            
            # Penalización por Ruina (Bankruptcy)
            # Si el agente pierde todo su capital, le damos un castigo severo
            if self.cash[i] <= 0:
                reward -= 1000 # Penalización extra
                # Opcional: done = True (si quieres acabar el episodio si uno quiebra)
            
            # Recompensa: Sharpe Ratio Proxy (PnL - Penalización Volatilidad)
            reward = total_pnl - Config.RISK_AVERSION * (abs(total_pnl)) 
            # Normalizar recompensa para estabilidad numérica
            rewards.append(reward / 1e6) 

        self.positions = next_positions
        next_obs = self._get_obs()
        
        return next_obs, np.array(rewards), done, {}