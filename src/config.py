
from dataclasses import dataclass

@dataclass
class Config:
    # Archivos
    path_futuros: str = "datos_futuros.csv"
    path_spot: str = "datos_energia.csv"

    # Selección de variable spot
    spot_variable: str = "PB_Nal"

    # Tipos de contrato
    contract_types: tuple = ("ELM", "ELS", "MTB", "DTB", "NTB")

    # Columnas
    date_col_futuros: str = "Fecha"
    tipo_col: str = "Tipo"
    mes_col: str = "Mes"
    anio_col: str = "Año"
    precio_col: str = "Precio"
    energia_fecha_col: str = "FechaHora"
    energia_var_col: str = "CodigoVariable"
    energia_valor_col: str = "Valor"
    energia_duracion_col: str = "CodigoDuracion"

    # Preprocesamiento
    fill_method: str = "ffill"
    start_date: str = None
    end_date: str = None

    # Estrategia
    hedge_notional: float = 1.0
    volume_scale: float = 1.0
    position_limit: float = 5.0
    tc_lambda: float = 1e-3

    # Roll-over
    rollover_commission: float = 10.0
    rollover_penalty: float = 0.001
    close_commission: float = 5.0

    # RL
    gamma: float = 0.99
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    tau: float = 5e-3
    batch_size: int = 256
    buffer_size: int = 200_000
    warmup_steps: int = 2_000
    train_steps: int = 50_000

    # Arquitectura
    actor_hidden: tuple = (256, 256)
    critic_hidden: tuple = (256, 256)
    act_noise_sigma: float = 0.10
    target_noise_sigma: float = 0.05
    target_noise_clip: float = 0.15

    device: str = "cuda"
    seed: int = 42