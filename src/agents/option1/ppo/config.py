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
            self.project_root / "src" / "models" / "option1" / "ppo",
        )
        object.__setattr__(
            self,
            "results_dir",
            self.project_root / "results" / "option1" / "ppo",
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

    comision_transaccion: float = 0.001  # 0.1% por transacción
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
    initial_capital_min: float = 5_000_000_000  # 5 mil millones COP


# =========================
# 4) Recompensas
# =========================
@dataclass(frozen=True)
class RewardConfig:
    """Hiperparámetros de la función de recompensa."""

    pnl_window_size: int = 30

    # Pesos
    w_pnl: float = 0.8
    w_risk: float = 0.50
    w_overhedge: float = 0.00
    w_transaction: float = 0.00
    w_opportunity: float = 0.00
    w_opportunity_expiry: float = 0.00
    w_coverage: float = 0.50
    w_capital_stress: float = 0.30
    w_margin_call: float = 0.20
    w_carry: float = 0.12
    w_invalid_action: float = 0.05

    # Escalas de normalización (COP / kWh)
    scale_pnl: float = 5e7
    scale_money: float = 1e7
    scale_tx: float = 1e6
    scale_kwh: float = 5e6
    scale_opportunity: float = 1e9
    scale_opportunity_expiry: float = 8e5
    scale_risk: float = 8e15
    scale_carry: float = 1e8


# =========================
# 5) Redes LSTM
# =========================
@dataclass(frozen=True)
class LSTMConfig:
    """Arquitectura recurrente para actor/crítico."""
    sequence_length: int = 30
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.0


# =========================
# 6) Agente DRL (PPO)
# =========================
@dataclass(frozen=True)
class PPOConfig:
    """Hiperparámetros de entrenamiento PPO."""

    # Optimizadores
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4

    # Descuento y ventaja
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO objective
    clip_eps: float = 0.20
    entropy_coef: float = 0.02
    value_coef: float = 0.50

    # Estabilidad
    max_grad_norm: float = 0.50
    target_kl: float = 0.03

    # Muestreo / actualización
    rollout_steps: int = 1024
    ppo_epochs: int = 10
    mini_batch_size: int = 256

    # Política Gaussiana continua
    action_std_init: float = 0.40
    action_std_min: float = 0.05
    action_std_decay: float = 0.9998


# =========================
# 7) General
# =========================
@dataclass(frozen=True)
class GeneralConfig:
    """Parámetros globales para reproducibilidad y entrenamiento."""

    seed: int = 20
    total_episodes: int = 200
    test_ratio: float = 0.09
    discretize_limit: float = 0.30

    # Inicio de iteraciones de negocio
    simulation_start_date: str = "2022-02-01"
    train_threshold: int = 250
    last_date_to_consider: str = "2026-01-30"

    # Logging opcional
    log_every: int = 10


@dataclass(frozen=True)
class ProjectConfig:
    """Contenedor principal de toda la configuración del proyecto."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    contract: ContractSpecConfig = field(default_factory=ContractSpecConfig)
    finance: FinanceConfig = field(default_factory=FinanceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)


# Instancia única de configuración para importar en otros módulos.
CONFIG = ProjectConfig()

# Crear carpetas de salida automáticamente al cargar el módulo.
CONFIG.paths.ensure_output_dirs()