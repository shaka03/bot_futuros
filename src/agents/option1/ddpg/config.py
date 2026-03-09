"""Configuración central del proyecto DRL para cobertura con futuros ELM.

Este módulo centraliza rutas, reglas de negocio y parámetros de entrenamiento
para evitar números mágicos en el resto del código.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


# =========================
# 1) Rutas (Paths)
# =========================
@dataclass(frozen=True)
class PathsConfig:
    """Rutas de entrada/salida del proyecto."""

    # Base del repositorio (ajustado a archivo actual)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4])

    # Datos de entrada
    data_dir: Path = field(init=False)
    dataset_sistema_file: Path = field(init=False)
    demanda_comprador_file: Path = field(init=False)
    fechas_transacciones_file: Path = field(init=False)
    precios_futuros_file: Path = field(init=False)
    datos_precios_file: Path = field(init=False)
    precios_liquidacion_file: Path = field(init=False)

    # Salidas
    model_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        data_dir = self.project_root / "data" / "3_gold"
        object.__setattr__(self, "data_dir", data_dir)

        object.__setattr__(self, "dataset_sistema_file", data_dir / "dataset_SISTEMA_ELM.csv")
        object.__setattr__(self, "demanda_comprador_file", data_dir / "demanda_COMPRADOR.csv")
        object.__setattr__(self, "fechas_transacciones_file", data_dir / "fechas_transacciones.csv")
        object.__setattr__(self, "precios_futuros_file", data_dir / "precios_FUTUROS.csv")
        object.__setattr__(self, "datos_precios_file", data_dir / "datos_PRECIOS.csv")
        object.__setattr__(self, "precios_liquidacion_file", data_dir / "precios_LIQUIDACION.csv")

        object.__setattr__(
            self,
            "model_dir",
            self.project_root / "src" / "models" / "option1" / "ddpg",
        )
        object.__setattr__(
            self,
            "results_dir",
            self.project_root / "results" / "option1",
        )

    def ensure_output_dirs(self) -> None:
        """Crea carpetas de salida si no existen."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 2) Especificaciones de Contratos
# =========================
@dataclass(frozen=True)
class ContractSpecConfig:
    """Parámetros del contrato ELM y límites de cobertura."""

    contract_type: str = "ELM"
    tamano_kwh: int = 360_000
    max_ordenes: int = 2_000
    curva_meses: int = 72
    bloque: str = "Dia"

    # Restricción principal del entorno
    max_horizon_months: int = 6


# =========================
# 3) Finanzas y Garantías
# =========================
@dataclass(frozen=True)
class FinanceConfig:
    """Parámetros de costos, margen y capital inicial dinámico."""

    comision_transaccion: float = 0.03
    umbral_margin_call: float = 0.75

    # Rangos de vencimiento en meses -> porcentaje de margen
    # clave: (inicio, fin), ambos inclusivos
    margenes_vencimiento: Dict[Tuple[int, int], float] = field(
        default_factory=lambda: {
            (0, 4): 0.234,
            (5, 8): 0.150,
            (9, 18): 0.130,
            (19, 24): 0.113,
            (24, 72): 0.100,
        }
    )

    # Hiperparámetro para cálculo de capital inicial dinámico
    factor_holgura: float = 3.0


# =========================
# 4) Recompensas
# =========================
@dataclass(frozen=True)
class RewardConfig:
    """Hiperparámetros de la función de recompensa."""

    lambda_riesgo: float = 1e-12
    lambda_penalizacion: float = 1e-5
    pnl_window_size: int = 30


# =========================
# 5) Redes LSTM
# =========================
@dataclass(frozen=True)
class LSTMConfig:
    """Arquitectura recurrente para actor/crítico."""

    sequence_length: int = 14
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1


# =========================
# 6) Agente DRL (DDPG)
# =========================
@dataclass(frozen=True)
class DDPGConfig:
    """Hiperparámetros de entrenamiento DDPG."""

    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_capacity: int = 300_000
    exploration_noise_std: float = 0.2
    exploration_noise_min_std: float = 0.03
    exploration_noise_decay: float = 0.997

@dataclass
class DDPGConfig:
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_capacity: int = 300_000
    exploration_noise_std: float = 0.20
    exploration_noise_min: float = 0.03
    exploration_noise_decay: float = 0.997


# =========================
# 7) Recompensas
# =========================
@dataclass
class RewardConfig:
    lambda_riesgo: float = 1e-12
    lambda_penalizacion: float = 1e-5
    lambda_turnover: float = 1e-6
    pnl_window_size: int = 30

# =========================
# 8) Contratos
# =========================
@dataclass
class ContractConfig:
    contract_type: str = "ELM"
    max_horizon_months: int = 6
    tamano_kwh: int = 1000
    max_ordenes: int = 300

    # Control de rebalanceo por step (evita churn)
    max_trade_fraction_per_step: float = 0.20
    min_trade_kwh: float = 1000.0


# =========================
# 9) General
# =========================
@dataclass(frozen=True)
class GeneralConfig:
    """Parámetros globales para reproducibilidad y entrenamiento."""
    seed: int = 42
    total_episodes: int = 400
    log_every: int = 10
    test_ratio: float = 0.10 # 90% entrenamiento, 10% prueba

    # Inicio de iteraciones de negocio
    simulation_start_date: str = "2022-02-01"

    # Pesos para el score de actualización de mejores modelos
    best_model_weight_reward: float = 0.5
    best_model_weight_pnl: float = 0.5


@dataclass(frozen=True)
class ProjectConfig:
    """Contenedor principal de toda la configuración del proyecto."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    contract: ContractSpecConfig = field(default_factory=ContractSpecConfig)
    finance: FinanceConfig = field(default_factory=FinanceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    ddpg: DDPGConfig = field(default_factory=DDPGConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ddpg: DDPGConfig = field(default_factory=DDPGConfig)


# Instancia única de configuración para importar en otros módulos.
CONFIG = ProjectConfig()

# Crear carpetas de salida automáticamente al cargar el módulo.
CONFIG.paths.ensure_output_dirs()