"""Entorno Gymnasium para cobertura con futuros de electricidad (ELM).

Incluye:
- Acción continua (6) discretizada a {-1, 0, 1}.
- Inventario por Nemotécnico (sin duplicidad por slot/contrato).
- Ciclo de vida del contrato: vencimiento -> cash settlement + liberación de margen.
- Mark-to-Market diario para contratos vigentes.
- Gestión de garantías: margin call, retiros y bancarrota (truncated).
- Recompensa media-varianza con penalizaciones por sobre-cobertura y compra duplicada.
- Rebalanceo controlado para permitir vender sin churn excesivo.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import CONFIG, ProjectConfig


@dataclass
class Position:
    """Representa una posición abierta por Nemotécnico."""

    nemotecnico: str
    month_slot: int  # 1..6
    fecha_apertura: pd.Timestamp
    fecha_vencimiento: pd.Timestamp
    quantity_contracts: int
    contract_size_kwh: int
    entry_price: float
    prev_price: float
    initial_margin_required: float
    maintenance_margin_required: float
    margin_balance: float


class ElectricityHedgingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Entorno personalizado de cobertura para entrenamiento DRL."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sequences_lstm: np.ndarray,
        futures_lookup: pd.DataFrame,              # MultiIndex (Fecha, Nemotecnico)
        nemotecnico_map_t1_t6: pd.DataFrame,       # index Fecha, cols Nemotecnico_t1..t6
        demand_aligned: pd.DataFrame,              # index Fecha
        precios_liquidacion: pd.DataFrame,         # cols: FechaVencimiento, Precio_COP/kWh_Dia
        initial_capital: float,
        config: ProjectConfig = CONFIG,
    ) -> None:
        super().__init__()
        self.config: ProjectConfig = config

        self.sequences_lstm = sequences_lstm.astype(np.float32)
        self.futures_lookup = futures_lookup
        self.nemotecnico_map = nemotecnico_map_t1_t6
        self.demand_aligned = demand_aligned
        self.precios_liquidacion = precios_liquidacion.copy()

        self.timeline: pd.DatetimeIndex = self.nemotecnico_map.index
        if not isinstance(self.timeline, pd.DatetimeIndex):
            raise TypeError("nemotecnico_map_t1_t6 debe tener DatetimeIndex en el índice.")

        # Espacios Gym
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.config.contract.max_horizon_months,),
            dtype=np.float32,
        )

        seq_len = self.sequences_lstm.shape[1]
        num_features = self.sequences_lstm.shape[2]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_len, num_features),
            dtype=np.float32,
        )

        # Estado interno
        self.initial_capital: float = float(initial_capital)
        self.current_capital: float = float(initial_capital)
        self.current_step: int = self.config.lstm.sequence_length

        self.inventory: Dict[str, Position] = {}
        self.margin_account_balance_total: float = 0.0
        self.pnl_history: Deque[float] = deque(maxlen=self.config.reward.pnl_window_size)

        # Estado extendido de cobertura (0/1) por slot 1..6
        self.coverage_state: np.ndarray = np.zeros(self.config.contract.max_horizon_months, dtype=np.float32)

        # Penalizaciones
        self.duplicate_buy_penalty_value: float = self.config.reward.lambda_penalizacion
        self.turnover_lambda: float = float(getattr(self.config.reward, "lambda_turnover", 0.0))

        # Parámetros de rebalanceo (fallback si no están en config)
        self.max_trade_fraction_per_step: float = float(
            getattr(self.config.contract, "max_trade_fraction_per_step", 0.20)
        )
        self.min_trade_kwh: float = float(getattr(self.config.contract, "min_trade_kwh", 1000.0))

        # Índice rápido de liquidación: FechaVencimiento -> Precio_COP/kWh_Dia
        self.liq_price_by_date: Dict[pd.Timestamp, float] = self._build_settlement_lookup(self.precios_liquidacion)

        self._validate_inputs()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Resetea el entorno a estado inicial."""
        super().reset(seed=seed)

        self.current_step = self.config.lstm.sequence_length
        self.current_capital = float(self.initial_capital)

        self.inventory.clear()
        self.margin_account_balance_total = 0.0
        self.pnl_history.clear()
        self.coverage_state[:] = 0.0

        obs = self._build_observation(self.current_step)
        info = {
            "capital_actual": self.current_capital,
            "sobre_cobertura_kwh": 0.0,
            "positions_open": 0,
            "covered_kwh_total": 0.0,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Ejecuta un paso temporal del entorno."""
        terminated = False
        truncated = False

        current_date = self.timeline[self.current_step]
        discrete_actions = self._discretize_action(action)

        pnl_delta = 0.0
        transaction_costs = 0.0
        duplicate_buy_penalty = 0.0
        margin_calls_cost = 0.0
        withdrawals = 0.0
        settlement_pnl = 0.0

        buys_count = 0
        sells_count = 0
        turnover_kwh = 0.0

        # --------------------------------------------------------------
        # 1) Procesamiento de acciones (rebalanceo controlado por slot)
        # --------------------------------------------------------------
        for month_slot, act in enumerate(discrete_actions, start=1):
            nem = self._get_nemotecnico_for_slot(current_date, month_slot)
            if nem is None:
                continue

            row = self._get_contract_row(current_date, nem)
            if row is None:
                continue

            # Validación de consistencia con mapeo 01..06
            expected_tipo = f"{self.config.contract.contract_type}_Vencimiento_{month_slot:02d}Meses"
            tipo_row = str(row.get("Tipo_Contrato", ""))
            if tipo_row != expected_tipo:
                continue

            price_today = float(row["Precio"])
            expiry_date = pd.to_datetime(row["FechaVencimientoContrato"])
            demand_col = f"Demanda_Comprador_Dia_{month_slot:02d}Meses_Adelante"
            expected_demand_kwh = self._get_expected_demand(current_date, demand_col)

            # límite de ajuste por step para evitar churn
            max_trade_kwh = self.max_trade_fraction_per_step * max(expected_demand_kwh, 1.0)

            # cobertura actual en ese slot y nem (0 o cobertura del inventario existente)
            current_cov_kwh_slot = 0.0
            if nem in self.inventory:
                pos_exist = self.inventory[nem]
                current_cov_kwh_slot = float(pos_exist.quantity_contracts * pos_exist.contract_size_kwh)

            # BUY / SELL deseado en kWh (acto discreto)
            desired_delta_kwh = 0.0
            if act == 1:
                desired_delta_kwh = max(0.0, expected_demand_kwh - current_cov_kwh_slot)  # acercarse a cobertura objetivo
            elif act == -1:
                desired_delta_kwh = -current_cov_kwh_slot  # cerrar parcial/total

            # clip por control de turnover
            delta_kwh = float(np.clip(desired_delta_kwh, -max_trade_kwh, max_trade_kwh))
            if abs(delta_kwh) < self.min_trade_kwh:
                continue

            # ---------------- BUY ----------------
            if delta_kwh > 0:
                if nem in self.inventory:
                    # si ya existe inventario del mismo nem, permitimos ampliar (sin duplicar objeto)
                    pos = self.inventory[nem]
                    add_qty = int(delta_kwh // self.config.contract.tamano_kwh)
                    if add_qty <= 0:
                        continue

                    margin_pct = self._get_margin_pct_for_slot(month_slot)
                    add_initial_margin = price_today * self.config.contract.tamano_kwh * add_qty * margin_pct
                    add_maintenance = add_initial_margin * self.config.finance.umbral_margin_call

                    notional = price_today * self.config.contract.tamano_kwh * add_qty
                    commission = notional * self.config.finance.comision_transaccion

                    required_cash = add_initial_margin + commission
                    if self.current_capital >= required_cash:
                        self.current_capital -= required_cash
                        transaction_costs += commission

                        # ampliar posición existente
                        pos.quantity_contracts += add_qty
                        pos.prev_price = price_today
                        pos.initial_margin_required += add_initial_margin
                        pos.maintenance_margin_required += add_maintenance
                        pos.margin_balance += add_initial_margin

                        self.coverage_state[month_slot - 1] = 1.0
                        buys_count += 1
                        turnover_kwh += add_qty * self.config.contract.tamano_kwh
                    else:
                        duplicate_buy_penalty += self.duplicate_buy_penalty_value
                    continue

                # apertura nueva
                qty_contracts = int(delta_kwh // self.config.contract.tamano_kwh)
                qty_contracts = max(0, min(qty_contracts, self.config.contract.max_ordenes))
                if qty_contracts == 0:
                    continue

                margin_pct = self._get_margin_pct_for_slot(month_slot)
                initial_margin = price_today * self.config.contract.tamano_kwh * qty_contracts * margin_pct
                maintenance_margin = initial_margin * self.config.finance.umbral_margin_call

                notional = price_today * self.config.contract.tamano_kwh * qty_contracts
                commission = notional * self.config.finance.comision_transaccion

                required_cash = initial_margin + commission
                if self.current_capital >= required_cash:
                    self.current_capital -= required_cash
                    transaction_costs += commission

                    self.inventory[nem] = Position(
                        nemotecnico=nem,
                        month_slot=month_slot,
                        fecha_apertura=current_date,
                        fecha_vencimiento=expiry_date,
                        quantity_contracts=qty_contracts,
                        contract_size_kwh=self.config.contract.tamano_kwh,
                        entry_price=price_today,
                        prev_price=price_today,
                        initial_margin_required=initial_margin,
                        maintenance_margin_required=maintenance_margin,
                        margin_balance=initial_margin,
                    )
                    self.coverage_state[month_slot - 1] = 1.0
                    buys_count += 1
                    turnover_kwh += qty_contracts * self.config.contract.tamano_kwh

            # ---------------- SELL (desarme controlado) ----------------
            elif delta_kwh < 0:
                if nem in self.inventory:
                    pos = self.inventory[nem]
                    close_qty = int(abs(delta_kwh) // self.config.contract.tamano_kwh)
                    close_qty = min(close_qty, pos.quantity_contracts)
                    if close_qty <= 0:
                        continue

                    close_pnl, released_margin, close_commission = self._close_position_partial(
                        nem=nem,
                        close_price=price_today,
                        quantity_to_close=close_qty,
                    )
                    pnl_delta += close_pnl
                    self.current_capital += released_margin
                    transaction_costs += close_commission
                    sells_count += 1
                    turnover_kwh += close_qty * self.config.contract.tamano_kwh

                    if nem not in self.inventory:
                        self.coverage_state[month_slot - 1] = 0.0
                # Si no existe, ignorar (no short naked)

        # --------------------------------------------------------------
        # 2) Ciclo de vida del contrato (obligatorio en cada t)
        #    2.1 Vencimiento -> cash settlement
        #    2.2 Si no vence -> MtM diario + garantías
        # --------------------------------------------------------------
        if not truncated:
            for nem in list(self.inventory.keys()):
                pos = self.inventory.get(nem)
                if pos is None:
                    continue

                # 2.1 Verificación de vencimiento
                if current_date >= pos.fecha_vencimiento:
                    # Precio de liquidación final (por fecha de vencimiento)
                    liq_price = self._get_settlement_price(pos.fecha_vencimiento)
                    if liq_price is None:
                        liq_price = pos.prev_price

                    final_pnl = (liq_price - pos.entry_price) * pos.contract_size_kwh * pos.quantity_contracts
                    settlement_pnl += final_pnl
                    self.current_capital += final_pnl

                    # Libera margen retenido
                    self.current_capital += pos.margin_balance

                    # Limpieza
                    del self.inventory[nem]
                    self.coverage_state[pos.month_slot - 1] = 0.0
                    continue

                # 2.2 MtM diario para contrato vigente
                today_price = self._get_price(current_date, nem)
                if today_price is None:
                    continue

                mtm_delta = (today_price - pos.prev_price) * pos.contract_size_kwh * pos.quantity_contracts
                pos.margin_balance += mtm_delta
                pos.prev_price = float(today_price)
                pnl_delta += mtm_delta

                # Gestión de garantías
                if pos.margin_balance < pos.maintenance_margin_required:
                    variation_margin = pos.initial_margin_required - pos.margin_balance
                    if self.current_capital >= variation_margin:
                        self.current_capital -= variation_margin
                        pos.margin_balance = pos.initial_margin_required
                        margin_calls_cost += variation_margin
                        pnl_delta -= variation_margin
                    else:
                        truncated = True
                        pnl_delta -= 1_000_000.0  # penalización severa bancarrota
                        break

                if pos.margin_balance > pos.initial_margin_required:
                    excess = pos.margin_balance - pos.initial_margin_required
                    pos.margin_balance = pos.initial_margin_required
                    self.current_capital += excess
                    withdrawals += excess
                    pnl_delta += excess

        # --------------------------------------------------------------
        # 3) Penalización sobre-cobertura + turnover
        # --------------------------------------------------------------
        overhedge_kwh = self._compute_overhedge_kwh(current_date)
        overhedge_penalty = self.config.reward.lambda_penalizacion * overhedge_kwh
        turnover_penalty = self.turnover_lambda * turnover_kwh

        # --------------------------------------------------------------
        # 4) Recompensa media-varianza
        # --------------------------------------------------------------
        pnl_step_total = pnl_delta + settlement_pnl
        pnl_mean_30 = float(np.mean(self.pnl_history)) if len(self.pnl_history) > 0 else 0.0
        risk_penalty = self.config.reward.lambda_riesgo * ((pnl_step_total - pnl_mean_30) ** 2)

        reward = (
            pnl_step_total
            - risk_penalty
            - overhedge_penalty
            - turnover_penalty
            - transaction_costs
            - duplicate_buy_penalty
        )

        self.pnl_history.append(float(pnl_step_total))
        self.margin_account_balance_total = float(sum(p.margin_balance for p in self.inventory.values()))

        # --------------------------------------------------------------
        # 5) Fin de episodio / salida
        # --------------------------------------------------------------
        self.current_step += 1
        if self.current_step >= len(self.timeline):
            terminated = True
            self.current_step = len(self.timeline) - 1

        obs = self._build_observation(self.current_step)
        covered_kwh_total = float(sum(p.quantity_contracts * p.contract_size_kwh for p in self.inventory.values()))

        info = {
            "capital_actual": float(self.current_capital),
            "sobre_cobertura_kwh": float(overhedge_kwh),
            "positions_open": len(self.inventory),
            "covered_kwh_total": covered_kwh_total,
            "pnl_delta_mtm": float(pnl_delta),
            "pnl_settlement": float(settlement_pnl),
            "transaction_costs": float(transaction_costs),
            "margin_calls_cost": float(margin_calls_cost),
            "withdrawals": float(withdrawals),
            "duplicate_buy_penalty": float(duplicate_buy_penalty),
            "turnover_kwh": float(turnover_kwh),
            "turnover_penalty": float(turnover_penalty),
            "buys_count": int(buys_count),
            "sells_count": int(sells_count),
            "margin_balance_total": float(self.margin_account_balance_total),
        }

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_observation(self, step: int) -> np.ndarray:
        """Construye observación y actualiza estado extendido de cobertura."""
        seq_len = self.config.lstm.sequence_length
        idx = step - seq_len + 1
        idx = max(0, min(idx, self.sequences_lstm.shape[0] - 1))

        obs = self.sequences_lstm[idx].copy()

        # Se asume que los últimos 6 features son Coverage_Mes_01..06.
        if obs.shape[1] >= self.config.contract.max_horizon_months:
            obs[:, -self.config.contract.max_horizon_months :] = self.coverage_state.reshape(1, -1)

        return obs

    def _discretize_action(self, action: np.ndarray) -> np.ndarray:
        """Discretiza acción continua con umbrales ±0.33."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        dim = self.config.contract.max_horizon_months
        if a.shape[0] != dim:
            raise ValueError(f"Acción inválida, se esperaban {dim} dimensiones y llegaron {a.shape[0]}.")

        out = np.zeros_like(a, dtype=np.int8)
        out[a <= -0.33] = -1
        out[a >= 0.33] = 1
        return out

    def _get_nemotecnico_for_slot(self, date: pd.Timestamp, month_slot: int) -> Optional[str]:
        """Devuelve Nemotecnico_t{slot} para la fecha actual."""
        col = f"Nemotecnico_t{month_slot}"
        if col not in self.nemotecnico_map.columns:
            return None
        v = self.nemotecnico_map.loc[date, col]
        if pd.isna(v):
            return None
        return str(v)

    def _get_contract_row(self, date: pd.Timestamp, nem: str) -> Optional[pd.Series]:
        """Obtiene registro de futuros por (Fecha, Nemotecnico)."""
        try:
            row = self.futures_lookup.loc[(date, nem)]
            if isinstance(row, pd.DataFrame):
                return row.iloc[0]
            return row
        except KeyError:
            return None

    def _get_price(self, date: pd.Timestamp, nem: str) -> Optional[float]:
        """Precio de cierre por (Fecha, Nemotecnico)."""
        row = self._get_contract_row(date, nem)
        if row is None:
            return None
        return float(row["Precio"])

    def _get_expected_demand(self, date: pd.Timestamp, demand_col: str) -> float:
        """Demanda esperada por slot para la fecha."""
        if demand_col not in self.demand_aligned.columns:
            return 0.0
        v = self.demand_aligned.loc[date, demand_col]
        return 0.0 if pd.isna(v) else float(v)

    def _get_margin_pct_for_slot(self, month_slot: int) -> float:
        """Obtiene porcentaje de margen según rango configurado."""
        for (start, end), pct in self.config.finance.margenes_vencimiento.items():
            if start <= month_slot <= end:
                return float(pct)
        return float(self.config.finance.margenes_vencimiento[(0, 4)])

    def _close_position(self, nem: str, close_price: float) -> Tuple[float, float, float]:
        """Cierre voluntario total de posición (compatibilidad)."""
        pos = self.inventory[nem]
        pnl_close = (close_price - pos.prev_price) * pos.contract_size_kwh * pos.quantity_contracts
        close_notional = close_price * pos.contract_size_kwh * pos.quantity_contracts
        close_commission = close_notional * self.config.finance.comision_transaccion
        released_margin = pos.margin_balance
        del self.inventory[nem]
        return float(pnl_close), float(released_margin), float(close_commission)

    def _close_position_partial(self, nem: str, close_price: float, quantity_to_close: int) -> Tuple[float, float, float]:
        """Cierre parcial FIFO simplificado de una posición por nem.

        Retorna:
        - pnl_cierre (desde prev_price para qty cerrada)
        - margen_liberado proporcional
        - comisión_cierre
        """
        pos = self.inventory[nem]
        qty = int(max(0, min(quantity_to_close, pos.quantity_contracts)))
        if qty == 0:
            return 0.0, 0.0, 0.0

        ratio = qty / pos.quantity_contracts

        pnl_close = (close_price - pos.prev_price) * pos.contract_size_kwh * qty
        close_notional = close_price * pos.contract_size_kwh * qty
        close_commission = close_notional * self.config.finance.comision_transaccion

        released_margin = pos.margin_balance * ratio
        released_initial_margin = pos.initial_margin_required * ratio
        released_maintenance_margin = pos.maintenance_margin_required * ratio

        pos.quantity_contracts -= qty
        pos.margin_balance -= released_margin
        pos.initial_margin_required -= released_initial_margin
        pos.maintenance_margin_required -= released_maintenance_margin

        if pos.quantity_contracts <= 0:
            del self.inventory[nem]

        return float(pnl_close), float(released_margin), float(close_commission)

    def _compute_overhedge_kwh(self, date: pd.Timestamp) -> float:
        """Calcula sobre-cobertura agregada en slots 1..6."""
        covered_by_slot: Dict[int, float] = {
            m: 0.0 for m in range(1, self.config.contract.max_horizon_months + 1)
        }
        for pos in self.inventory.values():
            covered_by_slot[pos.month_slot] += pos.quantity_contracts * pos.contract_size_kwh

        over = 0.0
        for m in range(1, self.config.contract.max_horizon_months + 1):
            demand_col = f"Demanda_Comprador_Dia_{m:02d}Meses_Adelante"
            expected = self._get_expected_demand(date, demand_col)
            over += max(0.0, covered_by_slot[m] - expected)
        return float(over)

    def _build_settlement_lookup(self, precios_liquidacion: pd.DataFrame) -> Dict[pd.Timestamp, float]:
        """Crea lookup de precio de liquidación por fecha de vencimiento."""
        df = precios_liquidacion.copy()
        if "FechaVencimiento" not in df.columns:
            return {}
        df["FechaVencimiento"] = pd.to_datetime(df["FechaVencimiento"])

        price_col = "Precio_COP/kWh_Dia"
        if price_col not in df.columns:
            return {}

        out: Dict[pd.Timestamp, float] = {}
        for _, r in df.iterrows():
            out[pd.Timestamp(r["FechaVencimiento"])] = float(r[price_col])
        return out

    def _get_settlement_price(self, expiry_date: pd.Timestamp) -> Optional[float]:
        """Obtiene precio de liquidación por fecha de vencimiento."""
        return self.liq_price_by_date.get(pd.Timestamp(expiry_date), None)

    def _validate_inputs(self) -> None:
        """Validaciones básicas de consistencia."""
        if self.sequences_lstm.ndim != 3:
            raise ValueError("sequences_lstm debe ser 3D: (num_samples, sequence_length, num_features).")

        required_fut_cols = {"Precio", "FechaVencimientoContrato", "Tipo_Contrato"}
        missing = required_fut_cols - set(self.futures_lookup.columns)
        if missing:
            raise KeyError(f"futures_lookup no contiene columnas requeridas: {missing}")

        for m in range(1, self.config.contract.max_horizon_months + 1):
            col = f"Nemotecnico_t{m}"
            if col not in self.nemotecnico_map.columns:
                raise KeyError(f"nemotecnico_map_t1_t6 debe incluir {col}")