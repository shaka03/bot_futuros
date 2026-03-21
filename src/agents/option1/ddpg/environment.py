"""Entorno Gymnasium para cobertura con futuros de electricidad (ELM).

Incluye:
- Acción continua (6) discretizada a {-1, 0, 1}.
- Inventario por Nemotécnico (sin duplicidad por slot/contrato).
- Ciclo de vida del contrato: vencimiento -> cash settlement + liberación de margen.
- Mark-to-Market diario para contratos vigentes.
- Gestión de garantías: margin call, retiros y bancarrota (truncated).
- Recompensa media-varianza con penalizaciones por sobre-cobertura, compra duplicada y costo de oportunidad por sub-cobertura.
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
    expected_demand_kwh: float = 0.0


class ElectricityHedgingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Entorno personalizado de cobertura para entrenamiento DRL."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sequences_lstm: np.ndarray,
        futures_lookup: pd.DataFrame,              # MultiIndex (Fecha, Nemotecnico)
        nemotecnico_map_t1_t6: pd.DataFrame,       # index Fecha, cols Nemotecnico_t1..t6
        demand_aligned: pd.DataFrame,              # index Fecha
        precios_liquidacion: pd.DataFrame,         # cols: FechaVencimiento, Precio_COP/kWh_bloque
        datos_precios: pd.DataFrame,
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

        # Índice rápido de liquidación: FechaVencimiento -> Precio_COP/kWh_bloque
        self.liq_price_by_date: Dict[pd.Timestamp, float] = self._build_settlement_lookup(self.precios_liquidacion)

        self.spot_price_lookup: Dict[pd.Timestamp, float] = self._build_spot_lookup(datos_precios)

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
            "pnl_mtm": 0.0,
            "pnl_settlement": 0.0,
            "opportunity_cost": 0.0,
            "risk_penalty": 0.0,
            "overhedge_penalty": 0.0,
            "current_date": str(self.timeline[self.current_step].date()),
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
        self._roll_positions_slots(current_date)
        discrete_actions = self._discretize_action(action)
        executed_actions = np.zeros_like(discrete_actions, dtype=np.int8)

        pnl_delta = 0.0
        transaction_costs = 0.0
        margin_calls_cost = 0.0
        withdrawals = 0.0
        settlement_pnl = 0.0
        opportunity_cost_expiry = 0.0

        contracts_net_step_by_slot = np.zeros(self.config.contract.max_horizon_months, dtype=np.int32)
        demand_to_cover_kwh_by_slot = np.zeros(self.config.contract.max_horizon_months, dtype=np.float64)

        # --------------------------------------------------------------
        # 1) Procesamiento de acciones (compras / ventas)
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
            demand_col = f"Demanda_Comprador_{self.config.contract.bloque}_{month_slot:02d}Meses_Adelante"
            expected_demand_kwh = self._get_expected_demand(current_date, demand_col)
            demand_to_cover_kwh_by_slot[month_slot - 1] = float(expected_demand_kwh)    

            # BUY
            if act == 1:
                # Si ya está cubierto con ese nemotécnico, ignorar y penalizar
                if nem in self.inventory:
                    continue

                # Cálculo de cantidad de contratos a comprar para cubrir demanda esperada del slot
                ## Obetner spot del día
                spot_price = self._get_spot_price(current_date)
                den = self.config.contract.tamano_kwh * max(price_today, 1e-12)
                qty_contracts = int(math.ceil((expected_demand_kwh * max(spot_price, 0.0)) / den))
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
                    self.coverage_state[month_slot - 1] = 1.0
                    executed_actions[month_slot - 1] = 1

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
                        expected_demand_kwh=expected_demand_kwh
                    )
                    contracts_net_step_by_slot[month_slot - 1] += int(qty_contracts)

            # SELL (cierre voluntario)
            elif act == -1:
                if nem in self.inventory:
                    qty_to_sell = self.inventory[nem].quantity_contracts
                    close_pnl, released_margin, close_commission = self._close_position(nem, price_today)
                    pnl_delta += close_pnl
                    self.current_capital += released_margin
                    self.current_capital -= close_commission
                    transaction_costs += close_commission
                    self.coverage_state[month_slot - 1] = 0.0
                    executed_actions[month_slot - 1] = -1
                    contracts_net_step_by_slot[month_slot - 1] -= int(qty_to_sell)
                # Si no existe, se ignora (no short naked)

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
                if current_date >= pos.fecha_vencimiento: #or current_date >= pd.to_datetime(self.config.general.last_date_to_consider):
                    # Precio de liquidación final (por fecha de vencimiento)
                    liq_price = self._get_settlement_price(pos.fecha_vencimiento)
                    spot_price = self._get_spot_price(pos.fecha_vencimiento)
                    if liq_price is None:
                        liq_price = pos.prev_price
                    
                    final_pnl = (spot_price - pos.prev_price) * pos.contract_size_kwh * pos.quantity_contracts
                    uncovered_demand_kwh = max(0.0, pos.expected_demand_kwh - (pos.quantity_contracts * pos.contract_size_kwh))
                    uncovered_cost = uncovered_demand_kwh * liq_price
                    settlement_pnl += final_pnl - uncovered_cost
                    self.current_capital += final_pnl - uncovered_cost

                    # Libera margen retenido
                    self.current_capital += pos.margin_balance

                    # Oportunidad al vencimiento (slot de la posición)
                    shortfall_kwh = uncovered_demand_kwh

                    liq_ref = float(liq_price) if liq_price is not None else float(pos.prev_price)
                    fut_ref = float(pos.entry_price)  # referencia simple
                    spread_exp = max(0.0, liq_ref - fut_ref)

                    opportunity_cost_expiry += spread_exp * shortfall_kwh

                    # Limpieza
                    del self.inventory[nem]
                    #self.coverage_state[pos.month_slot - 1] = 0.0
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
                        #pnl_delta -= variation_margin
                    else:
                        truncated = True
                        pnl_delta -= abs(variation_margin) * 2.0
                        break

                if pos.margin_balance > pos.initial_margin_required:
                    excess = pos.margin_balance - pos.initial_margin_required
                    pos.margin_balance = pos.initial_margin_required
                    self.current_capital += excess
                    withdrawals += excess
                    #pnl_delta += excess

        # --------------------------------------------------------------
        # 3) Penalización sobre-cobertura y sub-cobertura
        # --------------------------------------------------------------
        overhedge_kwh = self._compute_overhedge_kwh(current_date)
        #overhedge_penalty = self.config.reward.lambda_penalizacion * overhedge_kwh

        coverage_by_slot_kwh: Dict[int, float] = {
            m: 0.0 for m in range(1, self.config.contract.max_horizon_months + 1)
        }

        for pos in self.inventory.values():
            coverage_by_slot_kwh[pos.month_slot] += pos.quantity_contracts * pos.contract_size_kwh
        
        # Spot del día (promedio simple de precios disponibles t1..t6)
        #spot_candidates: List[float] = []
        #for m in range(1, self.config.contract.max_horizon_months + 1):
        #    nem_m = self._get_nemotecnico_for_slot(current_date, m)
        #    if nem_m is None:
        #        continue
        #    p_m = self._get_price(current_date, nem_m)
        #    if p_m is not None and np.isfinite(p_m):
        #        spot_candidates.append(float(p_m))
        #spot_price_ref = float(np.mean(spot_candidates)) if spot_candidates else 0.0
        spot_price_ref = self._get_spot_price_90(current_date)
        spot_price_curr = self._get_spot_price(current_date)

        # Carry por contango
        carry_cost = 0.0
        basis_by_slot = np.zeros(self.config.contract.max_horizon_months, dtype=np.float64)

        for month_slot in range(1, self.config.contract.max_horizon_months + 1):
            covered_kwh = float(coverage_by_slot_kwh[month_slot])
            if covered_kwh <= 0.0:
                continue

            nem_slot = self._get_nemotecnico_for_slot(current_date, month_slot)
            if nem_slot is None:
                continue

            fut_price = self._get_price(current_date, nem_slot)
            if fut_price is None:
                continue

            basis = float(fut_price) - float(spot_price_curr)
            basis_by_slot[month_slot - 1] = basis

            if basis > 0.0:
                carry_cost += basis * covered_kwh

        # Costo de oportunidad por shortfall:
        # Castiga no cubrir cuando ex-post el spot estuvo por encima del futuro del slot
        opportunity_cost = 0.0
        for month_slot in range(1, self.config.contract.max_horizon_months + 1):
            demand_col = f"Demanda_Comprador_{self.config.contract.bloque}_{month_slot:02d}Meses_Adelante"
            expected_demand_kwh = self._get_expected_demand(current_date, demand_col)
            covered_kwh = coverage_by_slot_kwh[month_slot]

            shortfall_kwh = max(0.0, expected_demand_kwh - covered_kwh)
            if shortfall_kwh <= 0.0:
                continue

            nem_slot = self._get_nemotecnico_for_slot(current_date, month_slot)
            if nem_slot is None:
                continue
            fut_price = self._get_price(current_date, nem_slot)
            if fut_price is None:
                continue

            if shortfall_kwh > 0:
                spread = max(0.0, spot_price_ref - float(fut_price))
                opportunity_cost += spread * shortfall_kwh

        # --------------------------------------------------------------
        # 4) Recompensa media-varianza
        # --------------------------------------------------------------
        pnl_step_total = pnl_delta + settlement_pnl
        pnl_mean_30 = float(np.mean(self.pnl_history)) if len(self.pnl_history) > 0 else 0.0
        downside = max(0.0, pnl_mean_30 - pnl_step_total) # solo penaliza si el paso actual es peor que la media histórica
        risk_penalty = (downside ** 2)
        
        # --------------------------------------------------------------
        # 5) Cálculo recompensa
        # --------------------------------------------------------------
        money_scale = max(float(self.config.reward.scale_money), 1.0)
        pnl_scale = max(float(self.config.reward.scale_pnl), 1.0)
        kwh_scale = max(float(self.config.reward.scale_kwh), 1.0)
        opp_scale = max(float(self.config.reward.scale_opportunity), 1.0)
        opp_expiry_scale = max(float(self.config.reward.scale_opportunity_expiry), 1.0)
        risk_scale = max(float(self.config.reward.scale_risk), 1.0)
        tx_scale = max(float(self.config.reward.scale_tx), 1.0)
        carry_scale = max(float(self.config.reward.scale_carry), 1.0)

        pnl_norm = pnl_step_total / pnl_scale
        risk_norm = risk_penalty / risk_scale
        overhedge_norm = overhedge_kwh / kwh_scale
        transaction_norm = transaction_costs / tx_scale
        opportunity_norm = opportunity_cost / opp_scale
        opportunity_expiry_norm = opportunity_cost_expiry / opp_expiry_scale
        carry_norm = carry_cost / carry_scale

        # Penalización de cobertura
        coverage_penalties: List[float] = []
        for m in range(1, self.config.contract.max_horizon_months + 1):
            demand_m = float(demand_to_cover_kwh_by_slot[m - 1])
            covered_m = float(coverage_by_slot_kwh[m])

            # ratio objetivo = 1.0 (ni sub ni sobre cobertura)
            ratio_m = covered_m / max(demand_m, 1.0)
            penalty_m = abs(1.0 - ratio_m)
            coverage_penalties.append(penalty_m)

        coverage_penalty = float(np.mean(coverage_penalties)) if coverage_penalties else 0.0

        # Penalización capital
        capital_ratio = self.current_capital / self.initial_capital
        stress_cap_penalty = max(0.0, 0.5 - capital_ratio)

        # Penalización margin call
        margin_call_norm = margin_calls_cost / money_scale

        reward = (
            self.config.reward.w_pnl * pnl_norm
            - self.config.reward.w_coverage * coverage_penalty
            - self.config.reward.w_risk * risk_norm
            - self.config.reward.w_overhedge * overhedge_norm
            - self.config.reward.w_transaction * transaction_norm
            - self.config.reward.w_opportunity * opportunity_norm
            - self.config.reward.w_opportunity_expiry * opportunity_expiry_norm
            - self.config.reward.w_capital_stress * stress_cap_penalty
            - self.config.reward.w_margin_call * margin_call_norm
            - self.config.reward.w_carry * carry_norm
        )
        if reward > 0:
            reward = min(reward, 5.0)   # cap ganancias
        elif truncated:
            reward = -1000.0 # bancarrota severa
        else:
            reward = max(reward, -10.0)  # permite castigo más fuerte

        self.pnl_history.append(float(pnl_step_total))
        self.margin_account_balance_total = float(sum(p.margin_balance for p in self.inventory.values()))

        # --------------------------------------------------------------
        # 6) Fin de episodio / salida
        # --------------------------------------------------------------
        self.current_step += 1
        if self.current_step >= len(self.timeline):
            terminated = True
            self.current_step = len(self.timeline) - 1

        obs = self._build_observation(self.current_step)
        info = {
            "capital_actual": float(self.current_capital),
            "sobre_cobertura_kwh": float(overhedge_kwh),
            "positions_open": len(self.inventory),
            "pnl_delta_mtm": float(pnl_delta),
            "pnl_settlement": float(settlement_pnl),
            "transaction_costs": float(transaction_costs),
            "margin_calls_cost": float(margin_calls_cost),
            "withdrawals": float(withdrawals),
            "risk_penalty": float(risk_penalty),
            "opportunity_cost": float(opportunity_cost),
            "spot_price": float(self._get_spot_price(current_date)),
            "future_price_t1": float(self._get_price(current_date, self._get_nemotecnico_for_slot(current_date, 1) or "")),
            "future_price_t2": float(self._get_price(current_date, self._get_nemotecnico_for_slot(current_date, 2) or "")),
            "future_price_t3": float(self._get_price(current_date, self._get_nemotecnico_for_slot(current_date, 3) or "")),
            "future_price_t4": float(self._get_price(current_date, self._get_nemotecnico_for_slot(current_date, 4) or "")),
            "future_price_t5": float(self._get_price(current_date, self._get_nemotecnico_for_slot(current_date, 5) or "")),
            "future_price_t6": float(self._get_price(current_date, self._get_nemotecnico_for_slot(current_date, 6) or "")),
            "margin_balance_total": float(self.margin_account_balance_total),
            "executed_disc_1": int(executed_actions[0]),
            "executed_disc_2": int(executed_actions[1]),
            "executed_disc_3": int(executed_actions[2]),
            "executed_disc_4": int(executed_actions[3]),
            "executed_disc_5": int(executed_actions[4]),
            "executed_disc_6": int(executed_actions[5]),
            "demand_to_cover_kwh_1": float(demand_to_cover_kwh_by_slot[0]),
            "demand_to_cover_kwh_2": float(demand_to_cover_kwh_by_slot[1]),
            "demand_to_cover_kwh_3": float(demand_to_cover_kwh_by_slot[2]),
            "demand_to_cover_kwh_4": float(demand_to_cover_kwh_by_slot[3]),
            "demand_to_cover_kwh_5": float(demand_to_cover_kwh_by_slot[4]),
            "demand_to_cover_kwh_6": float(demand_to_cover_kwh_by_slot[5]),
            "contracts_net_1": int(contracts_net_step_by_slot[0]),
            "contracts_net_2": int(contracts_net_step_by_slot[1]),
            "contracts_net_3": int(contracts_net_step_by_slot[2]),
            "contracts_net_4": int(contracts_net_step_by_slot[3]),
            "contracts_net_5": int(contracts_net_step_by_slot[4]),
            "contracts_net_6": int(contracts_net_step_by_slot[5]),
            "reward_total": float(reward),
            "reward_pnl_norm": float(pnl_norm),
            "reward_risk_norm": float(risk_norm),
            "reward_overhedge_norm": float(overhedge_norm),
            "reward_tx_norm": float(transaction_norm),
            "reward_opportunity_norm": float(opportunity_norm),
            "opportunity_cost_expiry": float(opportunity_cost_expiry),
            "reward_opportunity_expiry_norm": float(opportunity_expiry_norm),
            "reward_carry_norm": float(carry_norm),
            "coverage_penalty": float(coverage_penalty),
            "capital_ratio_norm": float(stress_cap_penalty),
            "current_date": str(current_date.date())
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

        n_cov = self.config.contract.max_horizon_months  # 6
        n_exp = 1
        n_cap = 1
        n_tail = n_cov + n_exp + n_cap
        n_feats = obs.shape[1]

        # Se asume que los últimos 6 features son Coverage_Mes_01..06.
        cov_start = n_feats - n_tail
        cov_end = cov_start + n_cov
        exp_idx = cov_end
        cap_idx = exp_idx + 1

        if cov_start >= 0:
            obs[:, cov_start:cov_end] = self.coverage_state.reshape(1, -1)

        # Cercanía vencimiento
        current_date = self.timeline[step]
        days_list = []
        for p in self.inventory.values():
            d = (p.fecha_vencimiento - current_date).days
            if d >= 0:
                days_list.append(d)

        if days_list:
            min_days = float(min(days_list))
        else:
            min_days = 999.0

        min_days_norm = min(min_days / 30.0, 1.0)  # [0,1]

        # Requiere 1 columnas extra al final del tensor de features
        if exp_idx >= 0:
            obs[:, exp_idx] = min_days_norm
        
        capital_ratio_curr = float(self.current_capital / self.initial_capital)
        capital_ratio_curr = float(np.clip(capital_ratio_curr, 0.0, 2.0))
        obs[:, cap_idx] = capital_ratio_curr

        return obs

    def _discretize_action(self, action: np.ndarray) -> np.ndarray:
        """Discretiza acción continua con umbrales ±0.33."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        dim = self.config.contract.max_horizon_months
        if a.shape[0] != dim:
            raise ValueError(f"Acción inválida, se esperaban {dim} dimensiones y llegaron {a.shape[0]}.")

        out = np.zeros_like(a, dtype=np.int8)
        out[a <= -self.config.general.discretize_limit] = -1
        out[a >= self.config.general.discretize_limit] = 1
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
        """Cierre voluntario de posición (no vencimiento).

        Retorna:
        - pnl_cierre (ajuste desde prev_price)
        - margen_liberado
        - comisión_cierre
        """
        pos = self.inventory[nem]
        pnl_close = (close_price - pos.prev_price) * pos.contract_size_kwh * pos.quantity_contracts
        close_notional = close_price * pos.contract_size_kwh * pos.quantity_contracts
        close_commission = close_notional * self.config.finance.comision_transaccion
        released_margin = pos.margin_balance
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
            demand_col = f"Demanda_Comprador_{self.config.contract.bloque}_{m:02d}Meses_Adelante"
            expected = self._get_expected_demand(date, demand_col)
            over += max(0.0, covered_by_slot[m] - expected)
        return float(over)

    def _build_settlement_lookup(self, precios_liquidacion: pd.DataFrame) -> Dict[pd.Timestamp, float]:
        """Crea lookup de precio de liquidación por fecha de vencimiento."""
        df = precios_liquidacion.copy()
        if "FechaVencimiento" not in df.columns:
            return {}
        df["FechaVencimiento"] = pd.to_datetime(df["FechaVencimiento"])

        price_col = f"Precio_COP/kWh_{self.config.contract.bloque}"
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
    
    def _roll_positions_slots(self, current_date: pd.Timestamp) -> None:
        self.coverage_state[:] = 0.0
        for pos in self.inventory.values():
            months_diff = (pos.fecha_vencimiento.year - current_date.year) * 12 + (
                pos.fecha_vencimiento.month - current_date.month
            )
            new_slot = max(1, min(self.config.contract.max_horizon_months, months_diff + 1))
            pos.month_slot = int(new_slot)

            if 1 <= pos.month_slot <= self.config.contract.max_horizon_months:
                # binario robusto
                self.coverage_state[pos.month_slot - 1] = 1.0
    
    def _build_spot_lookup(self, datos_precios: pd.DataFrame) -> Dict[pd.Timestamp, float]:
        """Construye lookup de precio spot diario desde datos_PRECIOS.csv."""
        df = datos_precios.copy()
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        price_col = f"Precio_COP/kWh_{self.config.contract.bloque}"
        if price_col not in df.columns:
            raise KeyError(f"Columna {price_col} no encontrada en datos_PRECIOS.csv")
        
        out: Dict[pd.Timestamp, float] = {}
        for _, row in df.iterrows():
            out[pd.Timestamp(row["Fecha"])] = float(row[price_col])
        return out
    
    def _get_spot_price(self, current_date: pd.Timestamp) -> float:
        """Obtiene el precio spot del día desde datos_PRECIOS.csv.
        
        Si no existe fecha exacta, busca la fecha más reciente anterior (ffill lógico).
        """
        if current_date in self.spot_price_lookup:
            return self.spot_price_lookup[current_date]
        
        available_dates = sorted(d for d in self.spot_price_lookup.keys() if d <= current_date)
        if available_dates:
            return self.spot_price_lookup[available_dates[-1]]
        
        return 0.0


    def _get_spot_price_90(self, current_date: pd.Timestamp) -> float:
        """Obtiene la media movil de 90 días de los precios spot del día desde datos_PRECIOS.csv.
        
        Si no existe fecha exacta, busca la media móvil más reciente anterior (ffill lógico).
        """
        if current_date in self.spot_price_lookup:
            # Media móvil simple de 90 días (si hay suficientes datos)
            window_size = 90
            prices = []
            for i in range(window_size):
                date = current_date - pd.Timedelta(days=i)
                if date in self.spot_price_lookup:
                    prices.append(self.spot_price_lookup[date])
            if prices:
                return sum(prices) / len(prices)
        
        # Buscar la fecha más reciente anterior
        available_dates = sorted(d for d in self.spot_price_lookup.keys() if d <= current_date)
        if available_dates:
            # Media móvil simple de 90 días para la fecha encontrada
            window_size = 90
            prices = []
            for i in range(window_size):
                date = available_dates[-1] - pd.Timedelta(days=i)
                if date in self.spot_price_lookup:
                    prices.append(self.spot_price_lookup[date])
            if prices:
                return sum(prices) / len(prices)
        
        return 0.0