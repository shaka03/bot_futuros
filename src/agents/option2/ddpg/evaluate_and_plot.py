"""Evaluación out-of-sample y visualización de resultados DDPG (Fase 6)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from config import CONFIG, ProjectConfig
from data_processor import DataProcessor
from ddpg_agent import DDPGAgent
from environment import ElectricityHedgingEnv


sns.set_theme(style="whitegrid", context="talk")


def _resolve_dirs(config: ProjectConfig) -> Tuple[Path, Path]:
    """Resuelve directorios de pesos y resultados."""
    weights_dir = Path(getattr(config.paths, "model_dir", "src/models/option2/ddpg"))
    results_dir = Path(getattr(config.paths, "results_dir", "results/option2/ddpg"))
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir, results_dir


def _split_out_of_sample(
    bundle,
    processor: DataProcessor,
    config: ProjectConfig,
):
    """Construye partición temporal out-of-sample para evaluación."""
    total_seq = bundle.lstm_sequences.shape[0]
    cut = max(1, int(total_seq * (1.0 - config.general.test_ratio)))

    seq_test = bundle.lstm_sequences[cut:].copy()

    # Índices de timeline compatibles con environment.reset() (current_step=sequence_length)
    full_timeline = bundle.nemotecnico_map_t1_t6.index
    seq_len = config.lstm.sequence_length

    # Mapeo aproximado: seq i corresponde al bloque [i : i+seq_len-1]
    # Para test, comenzamos desde índice temporal cut + seq_len - 1
    t_start = cut + seq_len - 1
    test_timeline = full_timeline[t_start:]
    if len(test_timeline) == 0:
        test_timeline = full_timeline[-(seq_len + 1) :]

    nem_test = bundle.nemotecnico_map_t1_t6.loc[test_timeline].copy()
    dem_test = bundle.demand_aligned.loc[test_timeline].copy()

    # Futuros: se filtra por fechas de test timeline
    fut = bundle.futures_lookup.reset_index()
    fut = fut[fut["Fecha"].isin(test_timeline)].copy()
    fut_test = fut.set_index(["Fecha", "Nemotecnico"]).sort_index()

    # Precios de liquidación (se usa completo)
    liq = processor.precios_liquidacion_df.copy() if processor.precios_liquidacion_df is not None else pd.DataFrame()
    prec = processor.datos_precios_df.copy() if processor.datos_precios_df is not None else pd.DataFrame()

    return seq_test, fut_test, nem_test, dem_test, liq, prec


def compute_spot_benchmark_costs(
    demand_aligned: pd.DataFrame,
    datos_precios_df: pd.DataFrame,
    config: ProjectConfig = CONFIG,
) -> pd.DataFrame:
    """Escenario base: compra 100% spot.

    Retorna DataFrame diario con:
    - Fecha
    - demanda_kwh
    - spot_price
    - spot_cost
    """
    prices = datos_precios_df.copy()
    prices["Fecha"] = pd.to_datetime(prices["Fecha"])

    # Columna spot (se asume la principal del CSV)
    if f"Precio_COP/kWh_{config.contract.bloque}" in prices.columns:
        spot_col = f"Precio_COP/kWh_{config.contract.bloque}"
    else:
        # fallback: primera columna numérica distinta de Fecha
        numeric_cols = [c for c in prices.columns if c != "Fecha" and pd.api.types.is_numeric_dtype(prices[c])]
        if not numeric_cols:
            raise KeyError("No se encontró columna de precio spot en datos_PRECIOS.csv")
        spot_col = numeric_cols[0]

    dem = demand_aligned.copy().reset_index().rename(columns={"index": "Fecha"})
    if "Fecha" not in dem.columns:
        dem = dem.rename(columns={dem.columns[0]: "Fecha"})

    # Demanda diaria base
    if f"Demanda_kWh_{config.contract.bloque}_Comprador" in dem.columns:
        dem_col = f"Demanda_kWh_{config.contract.bloque}_Comprador"
    else:
        demand_candidates = [c for c in dem.columns if "Demanda" in c and "Meses_Adelante" not in c]
        if not demand_candidates:
            raise KeyError("No se encontró columna de demanda diaria en demanda_aligned.")
        dem_col = demand_candidates[0]

    merged = dem.merge(prices[["Fecha", spot_col]], on="Fecha", how="left")
    merged[spot_col] = merged[spot_col].ffill()
    merged["spot_cost"] = merged[dem_col].astype(float) * merged[spot_col].astype(float)

    out = merged[["Fecha", dem_col, spot_col, "spot_cost"]].copy()
    out.columns = ["Fecha", "demanda_kwh", "spot_price", "spot_cost"]
    return out


def evaluate_agent_out_of_sample(config: ProjectConfig = CONFIG) -> Dict[str, pd.DataFrame]:
    """Ejecuta evaluación out-of-sample sin ruido y retorna tablas de resultados."""
    weights_dir, results_dir = _resolve_dirs(config)

    # Datos completos
    processor = DataProcessor(config)
    bundle = processor.get_agent_data("ELM")
    initial_capital = max(config.finance.initial_capital_min, bundle.dynamic_initial_capital)

    # Split test
    seq_test, fut_test, nem_test, dem_test, liq_test, prec_test = _split_out_of_sample(bundle, processor, config)

    # Entorno test
    env_test = ElectricityHedgingEnv(
        sequences_lstm=seq_test,
        futures_lookup=fut_test,
        nemotecnico_map_t1_t6=nem_test,
        demand_aligned=dem_test,
        precios_liquidacion=liq_test,
        datos_precios=prec_test,
        initial_capital=initial_capital,
        config=config,
    )

    # Agente (solo actor para inferencia)
    num_features = int(seq_test.shape[2])
    agent = DDPGAgent(
        num_features=num_features,
        action_dim=config.contract.max_horizon_months,
        config=config,
    )
    actor_path = weights_dir / "best_actor_ddpg.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"No se encontró modelo entrenado: {actor_path}")

    state_dict = torch.load(actor_path, map_location=agent.device)
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()

    # Benchmark spot (sobre período test)
    spot_daily = compute_spot_benchmark_costs(
        demand_aligned=dem_test,
        datos_precios_df=processor.datos_precios_df if processor.datos_precios_df is not None else pd.DataFrame(),
        config=config
    )

    # Rollout evaluación
    state, _ = env_test.reset()
    terminated, truncated = False, False

    logs: List[Dict[str, float | int | str | pd.Timestamp]] = []
    pnl_cum = 0.0

    while not (terminated or truncated):
        # acción continua sin ruido
        action = agent.select_action(state, add_noise=False)  # (6,)

        # para log de acciones discretizadas
        action_disc_raw = np.zeros_like(action, dtype=np.int8)
        action_disc_raw[action <= -config.general.discretize_limit] = -1
        action_disc_raw[action >= config.general.discretize_limit] = 1

        current_date = env_test.timeline[env_test.current_step]

        # info previa para mapear nemotécnicos por horizonte
        nem_by_slot = {}
        for m in range(1, config.contract.max_horizon_months + 1):
            col = f"Nemotecnico_t{m}"
            nem_by_slot[f"slot_{m}"] = (
                str(env_test.nemotecnico_map.loc[current_date, col])
                if col in env_test.nemotecnico_map.columns and pd.notna(env_test.nemotecnico_map.loc[current_date, col])
                else ""
            )

        next_state, reward, terminated, truncated, info = env_test.step(action)

        pnl_step = float(info.get("pnl_delta_mtm", 0.0)) + float(info.get("pnl_settlement", 0.0))
        pnl_cum += pnl_step

        # Spot del día
        spot_row = spot_daily.loc[spot_daily["Fecha"] == current_date]
        spot_price = float(spot_row["spot_price"].iloc[0]) if not spot_row.empty else np.nan

        # Cobertura efectiva total (kWh en inventario)
        covered_kwh = float(
            sum(pos.quantity_contracts * pos.contract_size_kwh for pos in env_test.inventory.values())
        )

        logs.append(
            {
                "Fecha": current_date,
                "reward": float(reward),
                "pnl_step": pnl_step,
                "pnl_acumulado": pnl_cum,
                "margen_inicial": float(info.get("margin_balance_total", 0.0)) + float(info.get("capital_actual", 0.0)),
                "capital_actual": float(info.get("capital_actual", np.nan)),
                "margin_balance_total": float(info.get("margin_balance_total", np.nan)),
                "sobre_cobertura_kwh": float(info.get("sobre_cobertura_kwh", 0.0)),
                "spot_price": spot_price,
                "future_price_t1": float(info.get("future_price_t1", np.nan)),
                "future_price_t2": float(info.get("future_price_t2", np.nan)),
                "future_price_t3": float(info.get("future_price_t3", np.nan)),
                "future_price_t4": float(info.get("future_price_t4", np.nan)),
                "future_price_t5": float(info.get("future_price_t5", np.nan)),
                "future_price_t6": float(info.get("future_price_t6", np.nan)),
                "covered_kwh_total": covered_kwh,
                "action_cont_1": float(action[0]),
                "action_cont_2": float(action[1]),
                "action_cont_3": float(action[2]),
                "action_cont_4": float(action[3]),
                "action_cont_5": float(action[4]),
                "action_cont_6": float(action[5]),
                "action_disc_raw_1": int(action_disc_raw[0]),
                "action_disc_raw_2": int(action_disc_raw[1]),
                "action_disc_raw_3": int(action_disc_raw[2]),
                "action_disc_raw_4": int(action_disc_raw[3]),
                "action_disc_raw_5": int(action_disc_raw[4]),
                "action_disc_raw_6": int(action_disc_raw[5]),
                "action_disc_exec_1": int(info.get("executed_disc_1", 0)),
                "action_disc_exec_2": int(info.get("executed_disc_2", 0)),
                "action_disc_exec_3": int(info.get("executed_disc_3", 0)),
                "action_disc_exec_4": int(info.get("executed_disc_4", 0)),
                "action_disc_exec_5": int(info.get("executed_disc_5", 0)),
                "action_disc_exec_6": int(info.get("executed_disc_6", 0)),
                "nem_slot_1": nem_by_slot["slot_1"],    
                "nem_slot_2": nem_by_slot["slot_2"],
                "nem_slot_3": nem_by_slot["slot_3"],
                "nem_slot_4": nem_by_slot["slot_4"],
                "nem_slot_5": nem_by_slot["slot_5"],
                "nem_slot_6": nem_by_slot["slot_6"],
                "margin_calls_cost": float(info.get("margin_calls_cost", 0.0)),
                "transaction_costs": float(info.get("transaction_costs", 0.0)),
                "liquidacion_mtm": float(info.get("pnl_delta_mtm", 0.0)),
                "liquidacion_vencimiento": float(info.get("pnl_settlement", 0.0)),
                "demanda_cubrir_kwh_1": float(info.get("demand_to_cover_kwh_1", 0.0)),
                "demanda_cubrir_kwh_2": float(info.get("demand_to_cover_kwh_2", 0.0)),
                "demanda_cubrir_kwh_3": float(info.get("demand_to_cover_kwh_3", 0.0)),
                "demanda_cubrir_kwh_4": float(info.get("demand_to_cover_kwh_4", 0.0)),
                "demanda_cubrir_kwh_5": float(info.get("demand_to_cover_kwh_5", 0.0)),
                "demanda_cubrir_kwh_6": float(info.get("demand_to_cover_kwh_6", 0.0)),
                "contratos_netos_1": int(info.get("contracts_net_1", 0)),
                "contratos_netos_2": int(info.get("contracts_net_2", 0)),
                "contratos_netos_3": int(info.get("contracts_net_3", 0)),
                "contratos_netos_4": int(info.get("contracts_net_4", 0)),
                "contratos_netos_5": int(info.get("contracts_net_5", 0)),
                "contratos_netos_6": int(info.get("contracts_net_6", 0)),
                "terminated": int(terminated),
                "truncated": int(truncated)
            }
        )

        state = next_state
        
    print(f"Eval ended | terminated={terminated} | truncated={truncated} | last_date={current_date}")

    eval_df = pd.DataFrame(logs)
    eval_df.to_csv(results_dir / "evaluation_rollout_ddpg.csv", index=False)
    spot_daily.to_csv(results_dir / "benchmark_spot_daily.csv", index=False)

    return {
        "eval_df": eval_df,
        "spot_daily": spot_daily,
        "results_dir_df": pd.DataFrame({"results_dir": [str(results_dir)]}),
    }


def build_financial_report(eval_df: pd.DataFrame, spot_daily: pd.DataFrame) -> pd.DataFrame:
    """Construye reporte financiero agregado."""
    # Costo estrategia DDPG aproximado = costo spot - pnl + costos operativos
    merged = eval_df.merge(spot_daily[["Fecha", "spot_cost"]], on="Fecha", how="left")
    merged["spot_cost"] = merged["spot_cost"].ffill().bfill()

    merged["strategy_cost_daily"] = (
        merged["spot_cost"]
        - merged["pnl_step"]
        + merged["transaction_costs"].fillna(0.0)
        #+ merged["margin_calls_cost"].fillna(0.0)
    )

    total_spot = float(merged["spot_cost"].sum())
    total_strategy = float(merged["strategy_cost_daily"].sum())
    ahorro_total = total_spot - total_strategy

    vol_spot = float(merged["spot_cost"].std(ddof=1))
    vol_strategy = float(merged["strategy_cost_daily"].std(ddof=1))
    vol_reduction = vol_spot - vol_strategy

    total_margin_calls = int((merged["margin_calls_cost"] > 0.0).sum())
    total_margin_calls_cost = float(merged["margin_calls_cost"].sum())

    pnl_mean = float(merged["pnl_step"].mean())
    pnl_std = float(merged["pnl_step"].std(ddof=1))
    sharpe = pnl_mean / pnl_std if pnl_std > 0 else 0.0

    report = pd.DataFrame(
        {
            "Ahorro_Total_COP": [ahorro_total],
            "Costo_Total_Spot_COP": [total_spot],
            "Costo_Total_Estrategia_DDPG_COP": [total_strategy],
            "Reduccion_Volatilidad_COP_std": [vol_reduction],
            "Volatilidad_Spot_std": [vol_spot],
            "Volatilidad_Estrategia_std": [vol_strategy],
            "Total_Margin_Calls": [total_margin_calls],
            "Total_Margin_Calls_COP": [total_margin_calls_cost],
            "Sharpe_PnL": [sharpe],
        }
    )
    return report, merged


def plot_learning_curve(results_dir: Path) -> None:
    """Gráfico 1: Curva de aprendizaje (reward y pnl por episodio)."""
    history_path = results_dir / "training_history_ddpg.csv"
    if not history_path.exists():
        return

    hist = pd.read_csv(history_path)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()

    ax1.plot(hist["episode"], hist["episode_reward"], label="Reward por episodio", color="#1f77b4", linewidth=2)
    ax2.plot(hist["episode"], hist["episode_pnl"], label="PnL por episodio", color="#ff7f0e", linewidth=2)

    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Reward", color="#1f77b4")
    ax2.set_ylabel("PnL (COP)", color="#ff7f0e")
    ax1.set_title("Curva de Aprendizaje DDPG")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(results_dir / "grafico_1_curva_aprendizaje.png", dpi=200)
    plt.close(fig)


def plot_monthly_cost_comparison(results_dir: Path, merged: pd.DataFrame) -> None:
    """Gráfico 2: Barras de costo mensual Spot vs DDPG."""
    df = merged.copy()
    df["Mes"] = pd.to_datetime(df["Fecha"]).dt.to_period("M").astype(str)

    monthly = (
        df.groupby("Mes", as_index=False)[["spot_cost", "strategy_cost_daily"]]
        .sum()
        .rename(columns={"spot_cost": "Escenario_Base_Spot", "strategy_cost_daily": "Estrategia_DDPG"})
    )

    monthly_long = monthly.melt(id_vars="Mes", var_name="Escenario", value_name="Costo_COP")

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(data=monthly_long, x="Mes", y="Costo_COP", hue="Escenario", ax=ax)
    ax.set_title("Comparativa de Costos Mensuales: Spot vs Estrategia DDPG")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Costo Total COP")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(results_dir / "grafico_2_costos_mensuales.png", dpi=200)
    plt.close()


def plot_coverage_vs_demand(results_dir: Path, merged: pd.DataFrame) -> None:
    """Gráfico 3: Área cobertura total vs demanda real diaria."""
    df = merged.copy()

    # demanda diaria de benchmark
    if "demanda_kwh" not in df.columns:
        # si no viene en merged, no graficar
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.fill_between(df["Fecha"], df["demanda_kwh"], alpha=0.35, label="Demanda kWh", color="#1f77b4")
    ax.fill_between(df["Fecha"], df["covered_kwh_total"], alpha=0.35, label="Cobertura kWh", color="#2ca02c")
    ax.plot(df["Fecha"], df["demanda_kwh"], color="#1f77b4", linewidth=1.5)
    ax.plot(df["Fecha"], df["covered_kwh_total"], color="#2ca02c", linewidth=1.5)

    ax.set_title("Cobertura Efectiva vs Demanda")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("kWh")
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "grafico_3_cobertura_vs_demanda.png", dpi=200)
    plt.close()


def plot_actions_heatmap(results_dir: Path, eval_df: pd.DataFrame) -> None:
    """Gráfico 4: Mapa de calor de decisiones por fecha y horizonte."""
    # Matriz: filas=Fecha, columnas=slot_1..slot_6, valores acción discreta
    mat = eval_df[
        ["Fecha", "action_disc_exec_1", "action_disc_exec_2", "action_disc_exec_3", "action_disc_exec_4", "action_disc_exec_5", "action_disc_exec_6"]
    ].copy()
    mat["Fecha"] = pd.to_datetime(mat["Fecha"]).dt.strftime("%Y-%m-%d")
    mat = mat.set_index("Fecha")

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.heatmap(
        mat.T,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Acción discreta (-1,0,1)"},
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Mapa de Calor de Acciones por Horizonte (1..6 meses)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Horizonte")
    plt.tight_layout()
    plt.savefig(results_dir / "grafico_4_mapa_calor_acciones.png", dpi=200)
    plt.close()


def evaluate_and_plot(config: ProjectConfig = CONFIG) -> None:
    """Pipeline completo de evaluación + reporte + visualización."""
    weights_dir, results_dir = _resolve_dirs(config)

    # 1-3) Evaluación y benchmark
    outs = evaluate_agent_out_of_sample(config)
    eval_df = outs["eval_df"]
    spot_daily = outs["spot_daily"]

    # Unir demanda benchmark para gráfico 3
    merged_report_base = eval_df.merge(spot_daily, on="Fecha", how="left")

    # 5) Reporte financiero
    report_df, merged_costs = build_financial_report(eval_df, spot_daily)
    merged_final = merged_costs.merge(spot_daily[["Fecha", "demanda_kwh"]], on="Fecha", how="left")
    merged_final["covered_kwh_total"] = eval_df["covered_kwh_total"].values

    report_df.to_csv(results_dir / "financial_report_ddpg.csv", index=False)

    # 4) Gráficos
    plot_learning_curve(results_dir)
    plot_monthly_cost_comparison(results_dir, merged_costs)
    plot_coverage_vs_demand(results_dir, merged_final)
    plot_actions_heatmap(results_dir, eval_df)

    print("Evaluación finalizada.")
    print(f"Resultados guardados en: {results_dir}")
    print(report_df.to_string(index=False))


#if __name__ == "__main__":
#    evaluate_and_plot(CONFIG)