import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import Config
from data_processor import DataProcessor

class EnergyHedgingEnv(gym.Env):
    def __init__(self, data, dates, train_mode=True):
        super(EnergyHedgingEnv, self).__init__()
        self.data = data
        self.dates = dates
        self.train_mode = train_mode
        self.n_agents = len(Config.CONTRACT_TYPES)
        self.dp = DataProcessor()
        
        # Acción: [Hedge_Ratio_Delta (-1 a 1), Roll_Prob (0 a 1)]
        # Hedge_Ratio_Delta: Cuánto cambiar la posición porcentualmente respecto al límite
        self.action_space = [spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32) 
                             for _ in range(self.n_agents)]
        
        # Estado: [Spot, M1, M2, DTM, Posicion_Actual]
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32) 
                                  for _ in range(self.n_agents)]
        
        # Empezamos en el índice 1 para tener un "ayer" (t-1) válido
        self.current_step = 1 
        self.positions = np.zeros(self.n_agents)
        
    def reset(self):
        # Reiniciar al día 1 (para tener historia del día 0)
        self.current_step = 1 
        self.positions = np.zeros(self.n_agents)
        return self._get_obs()

    def _get_obs(self):
        """
        Retorna el estado basado en información conocida (AYER).
        Evita el Look-ahead bias.
        """
        # El agente ve el cierre del paso anterior (t-1)
        yesterday_idx = self.current_step - 1
        yesterday_date = self.dates[yesterday_idx]
        
        obs_list = []
        for i, c_type in enumerate(Config.CONTRACT_TYPES):
            # Obtener datos de mercado de AYER
            market_data = self.dp.get_state_for_date(self.data, yesterday_date, c_type)
            # El estado incluye la posición que el agente TIENE actualmente (antes de decidir hoy)
            state = np.concatenate([market_data, [self.positions[i]]])
            obs_list.append(state)
            
        return obs_list

    def step(self, actions):
        # Fechas relevantes
        today_date = self.dates[self.current_step]
        yesterday_date = self.dates[self.current_step - 1]
        
        rewards = []
        # Verificar si es el final de los datos
        done = self.current_step >= len(self.dates) - 1
        
        portfolio_pnl = 0
        
        for i, action in enumerate(actions):
            c_type = Config.CONTRACT_TYPES[i]
            
            # --- 1. INTERPRETACIÓN DE ACCIONES ---
            # El agente decidió esto basado en lo que vio ayer
            hedge_delta = action[0] # Cambio deseado en la posición
            roll_prob = action[1]   # Decisión de Rollover
            
            # --- 2. OBTENER DATOS DE MERCADO ---
            # Datos de Ayer (t-1): Referencia de costo/entrada
            state_yesterday = self.dp.get_state_for_date(self.data, yesterday_date, c_type)
            spot_y, m1_y, m2_y, dtm_y = state_yesterday
            
            # Datos de Hoy (t): Referencia de valoración/salida
            state_today = self.dp.get_state_for_date(self.data, today_date, c_type)
            spot_t, m1_t, m2_t, dtm_t = state_today
            
            # --- 3. EJECUCIÓN Y REBALANCEO (Inicio del día) ---
            # Actualizamos la posición PRIMERO, porque el agente quiere estar expuesto HOY
            
            # Definir magnitud del cambio (ej. 10 contratos máx por paso)
            contracts_delta = hedge_delta * 10 
            
            # Costo de transacción: Se paga sobre el precio de ejecución.
            # Asumimos ejecución a la apertura de hoy ~= cierre de ayer (m1_y)
            cost = abs(contracts_delta * m1_y * Config.TRANSACTION_FEE)
            
            # Posición actualizada para enfrentar el día
            previous_position = self.positions[i]
            self.positions[i] += contracts_delta
            current_position = self.positions[i]
            
            # --- 4. CÁLCULO DE PNL (Durante el día) ---
            # Valoramos la NUEVA posición con el movimiento del mercado de hoy
            # PnL = Posición_Actual * (Precio_Hoy - Precio_Ayer)
            price_change = m1_t - m1_y
            daily_pnl = current_position * price_change
            
            # --- 5. LÓGICA DE ROLLOVERS Y VENCIMIENTOS ---
            roll_cost = 0
            settlement_pnl = 0
            
            # Caso A: Decisión de Roll-Over anticipado (si estamos cerca del vencimiento)
            # Solo permitimos roll si faltan pocos días (ej. < 5)
            if dtm_y <= 5 and roll_prob > 0.0:
                # El agente decide "saltar" al siguiente contrato para evitar el spot
                # Costo: Spread entre M2 y M1 (Roll Cost) + Comisiones extra
                spread = m2_y - m1_y
                # Asumimos que el costo es el spread adverso (simplificación)
                roll_cost = abs(current_position * spread * Config.ROLL_OVER_COST)
                
                # NOTA: En una simulación continua compleja, aquí cambiaríamos el ticker base.
                # Aquí penalizamos el costo para enseñar al agente que el roll no es gratis.
            
            # Caso B: Vencimiento (Liquidación Forzosa)
            # Si hoy DTM llega a 0 o menos, el contrato expira.
            if dtm_t <= 0:
                # Liquidación financiera contra Spot
                # La posición "muere" y se liquida por la diferencia entre Spot y Futuro
                # PnL Final = Posición * (Spot_Hoy - Futuro_Hoy)
                basis = spot_t - m1_t
                settlement_pnl = current_position * basis
                
                # Forzar cierre de posición (se queda en 0 para mañana)
                self.positions[i] = 0 
            
            # --- 6. PNL TOTAL Y RECOMPENSA ---
            total_step_pnl = daily_pnl - cost - roll_cost + settlement_pnl
            portfolio_pnl += total_step_pnl

            # ACTUALIZAR CAJA:
            self.cash += total_step_pnl

            # Opcional: Penalizar fuertemente si quiebra (cash < 0)
            if self.cash <= 0:
                done = True
                reward = -1000 # Castigo grande por quiebra
            
            # Recompensa ajustada al riesgo (Media-Varianza)
            reward = total_step_pnl - (Config.RISK_AVERSION * (total_step_pnl**2))
            rewards.append(reward)

        # Avanzar el reloj
        self.current_step += 1
        
        # La observación retornada es la de HOY (que será el "ayer" del siguiente paso)
        if not done:
            obs_next = self._get_obs() 
        else:
            obs_next = [np.zeros(5) for _ in range(self.n_agents)]

        info = {"pnl": portfolio_pnl}
        
        return obs_next, rewards, done, info