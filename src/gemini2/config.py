import torch
import os

class Config:
    # Rutas
    BASE_DIR = os.getcwd()
    DATA_PATH = os.path.join(BASE_DIR, "data/gold") # Ajusta si es necesario
    ENERGY_FILE = os.path.join(DATA_PATH, "datos_energia.csv")
    FUTURES_FILE = os.path.join(DATA_PATH, "datos_futuros.csv")
    IMG_DIR = os.path.join(BASE_DIR, "results/resultados_img_v2")

    # Definición del Entorno de Mercado
    CONTRACT_TYPES = ["ELM", "ELS", "MTB", "DTB", "NTB"]
    
    # Curva de Futuros: M1 (Frente), M2 (Siguiente), M3, etc.
    # El agente controlará posiciones en estos vencimientos simultáneamente.
    CURVE_SIZE = 3 
    
    # Capital y Costos
    INITIAL_CAPITAL = 1e10  # 10,000 Millones COP (para aguantar márgenes)
    TRANSACTION_FEE = 0.0005 # 0.05% por operación
    RISK_AVERSION = 0.8 # Lambda alto para forzar cobertura (Hedging)
    
    # Entrenamiento
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPISODES = 50
    BATCH_SIZE = 128
    BUFFER_CAPACITY = 100000
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3
    GAMMA = 0.99
    TAU = 0.005  # Soft update
    HIDDEN_DIM = 256  # Dimensiones ocultas para Actor y Critic
    TRAIN_SPLIT_PCT = 0.8  # 80% datos para entrenamiento
    
    # Dimensiones (se calculan dinámicamente, pero para referencia)
    # Estado por agente: [Spot_t-1, (Price_M1..M3)_t-1, (Pos_M1..M3)_t-1, (DTM_M1..M3), Cash]
    # Acción por agente: [Target_Ratio_M1, Target_Ratio_M2, Target_Ratio_M3] (-1 a 1)