import torch
import os

DATA_PATH = os.path.join(os.getcwd(), "data/gold")

class Config:
    # Rutas de archivos (ajustar según ubicación)
    
    ENERGY_FILE = os.path.join(DATA_PATH, "datos_energia.csv")
    FUTURES_FILE = os.path.join(DATA_PATH, "datos_futuros.csv")
    SAVE_DIR = os.path.join(os.getcwd(), "results/resultados_img_v1")

    # Configuración de Datos
    COD_VARIABLE_COL = "CodigoVariable" # Nombre de la columna para el código de variable
    COD_VARIABLE = ["PB_Nal"]  # Código de variable para precio spot nacional
    FECHA_SPOT_COL = "FechaHora"  # Nombre de la columna de fecha en datos de energía
    FECHA_FUT_COL = "Fecha"  # Nombre de la columna de fecha en datos de futuros
    PRECIO_COL = "Precio"  # Nombre de la columna de precio en datos de futuros

    # Configuración del Entorno
    CONTRACT_TYPES = ["ELM", "ELS", "MTB", "DTB", "NTB"]
    INITIAL_CAPITAL = 1e9  # 1000 Millones COP
    TRANSACTION_FEE = 0.001 # 0.1% por operación
    ROLL_OVER_COST = 0.002 # Costo extra por spread en rollover
    RISK_AVERSION = 0.5 # Factor lambda para penalizar varianza
    
    # Configuración de Entrenamiento
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    EPISODES = 50 # Ajustar según tiempo disponible
    MAX_STEPS = 200 # Pasos por episodio
    BATCH_SIZE = 64
    BUFFER_SIZE = 100000
    GAMMA = 0.99
    TAU = 0.005 # Soft update
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3
    HIDDEN_DIM = 256
    
    # Evitar Leakage
    TEST_PCT = 0.2 # 20% final para prueba