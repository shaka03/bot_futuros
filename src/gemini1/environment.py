import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import Config

class EnergyHedgingEnv(gym.Env):
    """
    Entorno de Simulación para Estrategias de Cobertura en Mercados de Energía.
    Cada agente representa una estrategia de cobertura para un tipo de contrato de futuros.
    """
    def __init__(self, data, dates, train_mode=True):
        """
        Inicializa el entorno con datos de mercado y configuración.
        
        Args:
            data (pd.DataFrame): DataFrame con datos de precios históricos.
            dates (list): Lista de fechas correspondientes a los datos.
            train_mode (bool): Modo de entrenamiento o evaluación.
        
        """
        super(EnergyHedgingEnv, self).__init__()
        self.data = data
        self.dates = dates
        self.train_mode = train_mode
        self.n_agents = len(Config.CONTRACT_TYPES)
        
        # Definir espacios de acción y observación
        # Acción: [Hedge_Ratio (-1 a 1), Roll_Decision (continuo, >0 ejecuta)] por agente
        self.action_space = [spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32) 
                             for _ in range(self.n_agents)]
        
        # Estado: [Spot, M1, M2, DTM, Current_Position]
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32) 
                                  for _ in range(self.n_agents)]
        
        self.current_step = 0
        self.positions = np.zeros(self.n_agents) # Posición actual en contratos M1
        self.cash = Config.INITIAL_CAPITAL
        
    def reset(self):
        """
        Resetea el entorno al estado inicial.
        
        Returns:
            list: Observaciones iniciales para cada agente.
        """
        self.current_step = 0
        self.positions = np.zeros(self.n_agents)
        self.cash = Config.INITIAL_CAPITAL
        return self._get_obs()

    def _get_obs(self):
        """
        Obtiene las observaciones actuales para cada agente.
        
        Returns:
            list: Lista de observaciones por agente.
        """
        current_date = self.dates[self.current_step]
        obs_list = []
        
        from data_processor import DataProcessor # Import local para evitar ciclo
        dp = DataProcessor() 
        
        for i, c_type in enumerate(Config.CONTRACT_TYPES):
            # Obtener datos de mercado (Spot, M1, M2, DTM)
            market_data = dp.get_state_for_date(self.data, current_date, c_type)
            # Concatenar con posición actual (Privada del agente)
            state = np.concatenate([market_data, [self.positions[i]]])
            obs_list.append(state)
            
        return obs_list

    def step(self, actions):
        """
        Ejecuta un paso en el entorno basado en las acciones de los agentes.
        
        Args:
            actions (list): Lista de acciones por agente.
        
        Returns:
            tuple: Observaciones siguientes, recompensas, estado de finalización, información adicional.
        """
        current_date = self.dates[self.current_step]
        rewards = []
        next_step = self.current_step + 1
        done = next_step >= len(self.dates) - 1
        
        portfolio_value_change = 0
        
        from data_processor import DataProcessor
        dp = DataProcessor()
        
        obs_next = []
        
        for i, action in enumerate(actions):
            c_type = Config.CONTRACT_TYPES[i]
            hedge_ratio_delta = action[0] # Cambio en posición
            roll_prob = action[1] # Decisión de Roll (> 0)
            
            # Estado actual
            market_data = dp.get_state_for_date(self.data, current_date, c_type)
            spot, m1, m2, dtm = market_data
            
            # --- Lógica de Negocio ---
            
            # 1. Calcular PnL diario de la posición mantenida
            # Si es el último paso, no hay next_date, usamos 0 cambio
            price_change = 0
            if not done:
                next_date = self.dates[next_step]
                next_market = dp.get_state_for_date(self.data, next_date, c_type)
                # Diferencia de precio del contrato M1
                price_change = next_market[1] - m1 
            
            # PnL por tenencia
            daily_pnl = self.positions[i] * price_change
            
            # 2. Costos de Transacción (Rebalanceo)
            # Definimos nueva posición objetivo
            # Simplificación: La acción determina cuánto cambiar la posición
            # Escalar acción a contratos reales (ej. max 100 contratos)
            contracts_delta = hedge_ratio_delta * 10 
            cost = abs(contracts_delta * m1 * Config.TRANSACTION_FEE)
            self.positions[i] += contracts_delta
            
            # 3. Lógica de Vencimiento y Roll-over
            # Si DTM es bajo (ej. < 5 días) y el agente decide hacer roll
            roll_cost = 0
            if dtm <= 5 and roll_prob > 0.0:
                # Cerrar M1, Abrir M2
                # Spread cost + Fee
                spread = m2 - m1
                roll_cost = abs(self.positions[i] * spread) + (abs(self.positions[i] * m2 * Config.ROLL_OVER_COST))
                # La posición se mantiene en cantidad, pero ahora rastrea M2 (que se volverá M1 en el siguiente reset lógico de datos)
                # En esta simulación simplificada, el "precio base" cambiaría, lo simulamos como costo
            
            # 4. Liquidación al Vencimiento
            settlement_pnl = 0
            if dtm == 0:
                # Liquidación financiera contra Spot
                # Ganancia/Pérdida = (Spot - PrecioFuturoEntrada) * Posición
                # Aquí modelamos el diferencial del día final
                basis = spot - m1
                settlement_pnl = self.positions[i] * basis
                self.positions[i] = 0 # Cerrar posición forzada
            
            total_step_pnl = daily_pnl - cost - roll_cost + settlement_pnl
            portfolio_value_change += total_step_pnl
            
            # Recompensa individual: Minimizar varianza local + Maximizar PnL (Cobertura eficiente)
            # Penalizamos movimientos bruscos y costos
            reward = total_step_pnl - (Config.RISK_AVERSION * (total_step_pnl**2))
            rewards.append(reward)

        self.current_step += 1
        if not done:
            obs_next = self._get_obs()
        else:
            # Dummy observation for done
            obs_next = [np.zeros(5) for _ in range(self.n_agents)]

        info = {"pnl": portfolio_value_change}
        
        return obs_next, rewards, done, info