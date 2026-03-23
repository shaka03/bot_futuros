"""Procesamiento de datos para cobertura con futuros ELM (PPO + LSTM).

Ajuste aplicado:
- El mapeo de nemotécnicos t+1..t+6 usa exclusivamente:
  ELM_Vencimiento_01Meses ... ELM_Vencimiento_06Meses
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import CONFIG, ProjectConfig


@dataclass(frozen=True)
class AgentDataBundle:
    """Bundle de datos listo para entorno/agente."""

    lstm_sequences: np.ndarray
    futures_lookup: pd.DataFrame
    demand_aligned: pd.DataFrame
    state_aligned_with_coverage: pd.DataFrame
    dynamic_initial_capital: float
    nemotecnico_map_t1_t6: pd.DataFrame
    scaler: RobustScaler
    scaled_feature_columns: List[str]
    coverage_feature_columns: List[str]
    expiry_feature_columns: List[str]
    capital_feature_columns: List[str]


class DataProcessor:
    """Clase principal de procesamiento de datos."""

    def __init__(self, config: ProjectConfig = CONFIG) -> None:
        self.config: ProjectConfig = config

        # Crudos
        self.dataset_sistema_df: Optional[pd.DataFrame] = None
        self.demanda_df: Optional[pd.DataFrame] = None
        self.fechas_transacciones_df: Optional[pd.DataFrame] = None
        self.precios_futuros_df: Optional[pd.DataFrame] = None
        self.datos_precios_df: Optional[pd.DataFrame] = None
        self.precios_liquidacion_df: Optional[pd.DataFrame] = None

        # Procesados
        self.futures_lookup_df: Optional[pd.DataFrame] = None  # index: (Fecha, Nemotecnico)
        self.state_aligned_df: Optional[pd.DataFrame] = None
        self.demand_aligned_df: Optional[pd.DataFrame] = None
        self.nemotecnico_map_df: Optional[pd.DataFrame] = None

        # Escalado / features
        self.scaler: Optional[RobustScaler] = None
        self.scaled_feature_columns: Optional[List[str]] = None
        self.coverage_feature_columns: List[str] = [
            f"Coverage_Mes_{i:02d}" for i in range(1, self.config.contract.max_horizon_months + 1)
        ]
        self.expiry_feature_columns: List[str] = [
            "MinDaysToExpiry_Open_Norm"
        ]
        self.capital_feature_columns: List[str] = ["CapitalRatio"]

    # ------------------------------------------------------------------
    # 1) Carga de datos
    # ------------------------------------------------------------------
    def load_data(self) -> None:
        """Carga los CSVs definidos en config.py y parsea fechas."""
        p = self.config.paths

        self.dataset_sistema_df = pd.read_csv(p.dataset_sistema_file)
        self.demanda_df = pd.read_csv(p.demanda_comprador_file)
        self.fechas_transacciones_df = pd.read_csv(p.fechas_transacciones_file)
        self.precios_futuros_df = pd.read_csv(p.precios_futuros_file)
        self.datos_precios_df = pd.read_csv(p.datos_precios_file)
        self.precios_liquidacion_df = pd.read_csv(p.precios_liquidacion_file)

        # Parseo de fechas
        self.dataset_sistema_df["Fecha"] = pd.to_datetime(self.dataset_sistema_df["Fecha"])
        self.demanda_df["Fecha"] = pd.to_datetime(self.demanda_df["Fecha"])
        self.fechas_transacciones_df["Fecha"] = pd.to_datetime(self.fechas_transacciones_df["Fecha"])
        self.precios_futuros_df["Fecha"] = pd.to_datetime(self.precios_futuros_df["Fecha"])
        self.precios_futuros_df["FechaVencimientoContrato"] = pd.to_datetime(
            self.precios_futuros_df["FechaVencimientoContrato"]
        )
        self.datos_precios_df["Fecha"] = pd.to_datetime(self.datos_precios_df["Fecha"])
        self.precios_liquidacion_df["FechaVencimiento"] = pd.to_datetime(
            self.precios_liquidacion_df["FechaVencimiento"]
        )

        # Orden
        self.dataset_sistema_df.sort_values("Fecha", inplace=True)
        self.demanda_df.sort_values("Fecha", inplace=True)
        self.fechas_transacciones_df.sort_values("Fecha", inplace=True)
        self.precios_futuros_df.sort_values(["Fecha", "Nemotecnico"], inplace=True)
        self.datos_precios_df.sort_values("Fecha", inplace=True)
        self.precios_liquidacion_df.sort_values("FechaVencimiento", inplace=True)

    # ------------------------------------------------------------------
    # 2) Transformación y alineación temporal (sin leakage)
    # ------------------------------------------------------------------
    def prepare_futures_lookup(self, agent_id: str = "ELM") -> pd.DataFrame:
        """Mantiene futuros en largo y crea MultiIndex (Fecha, Nemotecnico)."""
        self._check_loaded()

        fut = self.precios_futuros_df.copy()
        fut = fut[fut["Tipo"].astype(str).str.upper() == agent_id.upper()].copy()

        fut.set_index(["Fecha", "Nemotecnico"], inplace=True)
        fut.sort_index(inplace=True)

        self.futures_lookup_df = fut
        return fut

    def align_to_transaction_dates(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Alinea dataset_SISTEMA_ELM y demanda_COMPRADOR al índice maestro de transacciones.

        Regla estricta:
        - Solo forward fill (ffill)
        - No bfill, no interpolación
        """
        self._check_loaded()

        start_date = pd.to_datetime(self.config.general.simulation_start_date)

        tx = (
            self.fechas_transacciones_df.loc[self.fechas_transacciones_df["Fecha"] >= start_date, ["Fecha"]]
            .drop_duplicates()
            .sort_values("Fecha")
            .set_index("Fecha")
        )

        st = self.dataset_sistema_df.copy()
        st = st[st["Fecha"] >= start_date].copy().set_index("Fecha").sort_index()
        st_aligned = st.reindex(tx.index).ffill()

        dm = self.demanda_df.copy()
        dm = dm[dm["Fecha"] >= start_date].copy().set_index("Fecha").sort_index()
        dm_aligned = dm.reindex(tx.index).ffill()

        self.state_aligned_df = st_aligned
        self.demand_aligned_df = dm_aligned

        return st_aligned, dm_aligned

    def build_nemotecnico_map_t1_t6(self, agent_id: str = "ELM") -> pd.DataFrame:
        """Genera tabla auxiliar de nemotécnicos t+1..t+6 por fecha de transacción.

        Ajuste importante:
        - Usa Tipo_Contrato = ELM_Vencimiento_01..06Meses.
        """
        if self.futures_lookup_df is None:
            self.prepare_futures_lookup(agent_id=agent_id)
        assert self.futures_lookup_df is not None

        if self.state_aligned_df is None:
            self.align_to_transaction_dates()
        assert self.state_aligned_df is not None

        tx_dates = self.state_aligned_df.index
        rows: List[Dict[str, object]] = []

        fut = self.futures_lookup_df.reset_index()
        fut = fut[fut["Tipo"].astype(str).str.upper() == agent_id.upper()].copy()

        # AJUSTE: horizonte 1..6 usando 01..06
        valid_tipo_contrato = {
            f"{agent_id}_Vencimiento_{i:02d}Meses": i
            for i in range(1, self.config.contract.max_horizon_months + 1)
        }

        fut["month_ahead"] = fut["Tipo_Contrato"].map(valid_tipo_contrato)
        fut = fut[fut["month_ahead"].notna()].copy()
        fut["month_ahead"] = fut["month_ahead"].astype(int)

        fut = fut.sort_values(["Fecha", "month_ahead", "Nemotecnico"]).drop_duplicates(
            subset=["Fecha", "month_ahead"], keep="first"
        )

        grouped = fut.groupby("Fecha")

        for dt in tx_dates:
            out: Dict[str, object] = {"Fecha": dt}
            if dt in grouped.groups:
                day_df = grouped.get_group(dt)
                for m in range(1, self.config.contract.max_horizon_months + 1):
                    r = day_df.loc[day_df["month_ahead"] == m, "Nemotecnico"]
                    out[f"Nemotecnico_t{m}"] = str(r.iloc[0]) if not r.empty else np.nan
            else:
                for m in range(1, self.config.contract.max_horizon_months + 1):
                    out[f"Nemotecnico_t{m}"] = np.nan
            rows.append(out)

        nem_map = pd.DataFrame(rows).set_index("Fecha").sort_index()
        #nem_map = nem_map.ffill()  # solo ffill

        self.nemotecnico_map_df = nem_map
        return nem_map

    # ------------------------------------------------------------------
    # 4) Capital inicial dinámico
    # ------------------------------------------------------------------
    def calculate_dynamic_capital(self) -> float:
        """Calcula capital inicial dinámico en COP según fórmula de negocio."""
        self._check_loaded()

        demand_col = f"Demanda_kWh_{self.config.contract.bloque}_Comprador"
        if demand_col not in self.demanda_df.columns:
            raise KeyError(f"No existe columna requerida: {demand_col}")

        max_demanda = float(pd.to_numeric(self.demanda_df[demand_col], errors="coerce").max())
        if np.isnan(max_demanda):
            raise ValueError("Demanda máxima inválida (NaN).")

        max_price = float(pd.to_numeric(self.precios_futuros_df["Precio"], errors="coerce").max())
        if np.isnan(max_price):
            raise ValueError("Precio máximo inválido (NaN).")

        tamano = self.config.contract.tamano_kwh
        max_contracts = int(np.ceil(max_demanda / tamano))
        margin_0_4 = self.config.finance.margenes_vencimiento[(0, 4)]
        factor = self.config.finance.factor_holgura

        capital = (max_contracts * max_price * tamano * margin_0_4) * factor
        return float(capital)

    # ------------------------------------------------------------------
    # 5) Normalización + features de cobertura
    # ------------------------------------------------------------------
    def fit_transform_state_with_coverage(
        self,
        state_aligned: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Escala solo variables de estado y agrega 6 flags binarios Coverage_Mes_01..06."""
        if state_aligned is None:
            if self.state_aligned_df is None:
                raise ValueError("Ejecute align_to_transaction_dates() primero.")
            state_aligned = self.state_aligned_df

        df = state_aligned.copy()

        for c in self.coverage_feature_columns:
            if c not in df.columns:
                df[c] = 0.0
        
        for c in self.expiry_feature_columns:
            if c not in df.columns:
                df[c] = 0.0
        
        for c in self.capital_feature_columns:
            if c not in df.columns:
                df[c] = 0.0

        test_ratio = float(self.config.general.test_ratio)
        split_idx = int(len(df) * (1.0 - test_ratio))
        
        scale_cols = self._select_state_columns_for_scaling(df)

        scaler = RobustScaler()
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        df_train.loc[:, scale_cols] = scaler.fit_transform(df_train[scale_cols])
        df_test.loc[:, scale_cols] = scaler.transform(df_test[scale_cols])

        df = pd.concat([df_train, df_test], axis=0).sort_index()

        self.scaler = scaler
        self.scaled_feature_columns = scale_cols

        return df

    # ------------------------------------------------------------------
    # 6) Secuencias LSTM
    # ------------------------------------------------------------------
    def generate_lstm_sequences(self, state_with_coverage_df: pd.DataFrame) -> np.ndarray:
        """Genera secuencias (num_samples, sequence_length, num_features)."""
        if self.scaled_feature_columns is None:
            raise ValueError("Primero ejecute fit_transform_state_with_coverage().")

        feature_cols = (
            self.scaled_feature_columns
            + self.coverage_feature_columns
            + self.expiry_feature_columns
            + self.capital_feature_columns
        )
        values = state_with_coverage_df[feature_cols].to_numpy(dtype=np.float32)

        seq_len = self.config.lstm.sequence_length
        n_rows, n_features = values.shape
        if n_rows < seq_len:
            raise ValueError(
                f"No hay suficientes filas para sequence_length={seq_len}. Disponibles={n_rows}"
            )

        n_samples = n_rows - seq_len + 1
        seq = np.empty((n_samples, seq_len, n_features), dtype=np.float32)

        for i in range(n_samples):
            seq[i] = values[i : i + seq_len]

        return seq

    # ------------------------------------------------------------------
    # 7) Estructura final por agente
    # ------------------------------------------------------------------
    def get_agent_data(self, agent_id: str = "ELM") -> AgentDataBundle:
        """Pipeline completo y retorno estructurado."""
        self.load_data()
        self.prepare_futures_lookup(agent_id=agent_id)
        self.align_to_transaction_dates()
        self.build_nemotecnico_map_t1_t6(agent_id=agent_id)

        assert self.state_aligned_df is not None
        assert self.demand_aligned_df is not None
        assert self.futures_lookup_df is not None
        assert self.nemotecnico_map_df is not None

        state_with_cov = self.fit_transform_state_with_coverage(self.state_aligned_df)
        sequences = self.generate_lstm_sequences(state_with_cov)
        capital = self.calculate_dynamic_capital()

        assert self.scaler is not None
        assert self.scaled_feature_columns is not None

        return AgentDataBundle(
            lstm_sequences=sequences,
            futures_lookup=self.futures_lookup_df,
            demand_aligned=self.demand_aligned_df,
            state_aligned_with_coverage=state_with_cov,
            dynamic_initial_capital=capital,
            nemotecnico_map_t1_t6=self.nemotecnico_map_df,
            scaler=self.scaler,
            scaled_feature_columns=self.scaled_feature_columns,
            coverage_feature_columns=self.coverage_feature_columns,
            expiry_feature_columns=self.expiry_feature_columns,
            capital_feature_columns=self.capital_feature_columns,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_loaded(self) -> None:
        req = [
            self.dataset_sistema_df,
            self.demanda_df,
            self.fechas_transacciones_df,
            self.precios_futuros_df,
            self.datos_precios_df,
            self.precios_liquidacion_df,
        ]
        if any(x is None for x in req):
            raise RuntimeError("Datos no cargados. Ejecute load_data().")

    @staticmethod
    def _is_numeric(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s)

    def _select_state_columns_for_scaling(self, df: pd.DataFrame) -> List[str]:
        """Selecciona columnas de estado a escalar (excluye operativas y flags)."""
        numeric_cols = [c for c in df.columns if self._is_numeric(df[c])]

        excluded_patterns = [
            #"Demanda_Comprador",
            #f"Demanda_kWh_{self.config.contract.bloque}_Comprador",
            "Coverage_Mes_",
            "MinDaysToExpiry",
            "CapitalRatio"
        ]

        selected: List[str] = []
        for c in numeric_cols:
            if any(pat in c for pat in excluded_patterns):
                continue
            selected.append(c)

        if not selected:
            raise ValueError("No se encontraron columnas válidas para escalado.")
        return selected


#if __name__ == "__main__":
#    dp = DataProcessor(CONFIG)
#    bundle = dp.get_agent_data("ELM")
#    print("OK - Bundle listo")
#    print("Secuencias:", bundle.lstm_sequences.shape)
#    print("Capital dinámico COP:", round(bundle.dynamic_initial_capital, 2))