"""Script principal de entrenamiento DDPG (Fase 5)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from config import CONFIG, ProjectConfig
from data_processor import DataProcessor
from ddpg_agent import DDPGAgent
from environment import ElectricityHedgingEnv


def _resolve_training_hyperparams(config: ProjectConfig) -> Tuple[int, int]:
    """Extrae total_episodes y log_every con fallback seguro."""
    total_episodes = int(getattr(config.general, "total_episodes", 200))
    log_every = int(getattr(config.general, "log_every", 10))
    return total_episodes, log_every


def _resolve_output_dirs(config: ProjectConfig) -> Tuple[Path, Path]:
    """Resuelve directorios para pesos y resultados."""
    # Pesos (ej. src/agents/option1/ddpg/)
    if hasattr(config.paths, "model_dir"):
        weights_dir = Path(config.paths.model_dir)
    else:
        weights_dir = Path("src/models/option1/ddpg")

    # Resultados (ej. results/option1/)
    if hasattr(config.paths, "results_dir"):
        results_dir = Path(config.paths.results_dir)
    else:
        results_dir = Path("results/option1")

    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir, results_dir

def _split_train_bundle(bundle, processor, config):
    """Construye partición temporal de entrenamiento (1 - test_ratio)."""
    test_ratio = float(config.general.test_ratio)
    total_seq = bundle.lstm_sequences.shape[0]
    cut = max(1, int(total_seq * (1.0 - test_ratio)))  # train seq = [0:cut]

    seq_train = bundle.lstm_sequences[:cut].copy()

    full_timeline = bundle.nemotecnico_map_t1_t6.index
    seq_len = config.lstm.sequence_length

    # timeline compatible con environment (current_step = seq_len)
    # para train: desde (seq_len-1) hasta (cut + seq_len - 2) inclusive
    t_start = seq_len - 1
    t_end = min(len(full_timeline), cut + seq_len - 1)
    train_timeline = full_timeline[t_start:t_end]

    nem_train = bundle.nemotecnico_map_t1_t6.loc[train_timeline].copy()
    dem_train = bundle.demand_aligned.loc[train_timeline].copy()

    fut = bundle.futures_lookup.reset_index()
    fut = fut[fut["Fecha"].isin(train_timeline)].copy()
    fut_train = fut.set_index(["Fecha", "Nemotecnico"]).sort_index()

    liq = processor.precios_liquidacion_df.copy() if processor.precios_liquidacion_df is not None else pd.DataFrame()
    prec = processor.datos_precios_df.copy() if processor.datos_precios_df is not None else pd.DataFrame()

    return seq_train, fut_train, nem_train, dem_train, liq, prec


def train_ddpg_agent(config: ProjectConfig = CONFIG) -> Dict[str, List[float]]:
    """Orquesta entrenamiento completo DDPG."""

    # ------------------------------------------------------------------
    # 1) Inicialización de módulos
    # ------------------------------------------------------------------
    processor = DataProcessor(config)
    bundle = processor.get_agent_data("ELM")
    initial_capital = max(config.finance.initial_capital_min, bundle.dynamic_initial_capital)

    seq_train, fut_train, nem_train, dem_train, liq_train, prec_train = _split_train_bundle(bundle, processor, config)

    env = ElectricityHedgingEnv(
        sequences_lstm=seq_train,
        futures_lookup=fut_train,
        nemotecnico_map_t1_t6=nem_train,
        demand_aligned=dem_train,
        precios_liquidacion=liq_train,
        datos_precios=prec_train,
        initial_capital=initial_capital,
        config=config,
    )

    num_features = int(seq_train.shape[2])
    agent = DDPGAgent(
        num_features=num_features,
        action_dim=config.contract.max_horizon_months,  # 6
        config=config,
    )

    total_episodes, log_every = _resolve_training_hyperparams(config)
    weights_dir, results_dir = _resolve_output_dirs(config)

    # ------------------------------------------------------------------
    # 2) Métricas de entrenamiento
    # ------------------------------------------------------------------
    episode_rewards: List[float] = []
    episode_pnls: List[float] = []
    margin_calls_count: List[int] = []
    overhedging_penalties: List[float] = []
    episode_times_sec: List[float] = []
    episode_tx_costs: List[float] = []
    episode_ahorros: List[float] = []

    best_reward = -np.inf

    train_no_improvement_count = 0
    train_threshold = int(getattr(config.general, "train_threshold", 20))

    # ------------------------------------------------------------------
    # 3) Bucle de entrenamiento por episodios
    # ------------------------------------------------------------------
    for episode in range(1, total_episodes + 1):
        t0 = time.perf_counter()

        state, info = env.reset()
        terminated = False
        truncated = False

        ep_reward = 0.0
        ep_pnl = 0.0
        ep_margin_calls = 0
        ep_overhedge = 0.0
        ep_coverage_penalty = 0.0
        ep_pnl_norm = 0.0
        ep_risk_norm = 0.0
        ep_overhedge_norm = 0.0
        ep_transaction_norm = 0.0
        ep_duplicate_norm = 0.0
        ep_opportunity_norm = 0.0
        ep_opportunity_expiry_norm = 0.0
        ep_capital_ratio = 0.0
        ep_carry_norm = 0.0
        conteo = 0

        ep_tx_cost = 0.0
        ep_ahorro = 0.0

        # ------------------------------------------------------------------
        # 4) Inner loop: interacción + aprendizaje
        # ------------------------------------------------------------------
        while not (terminated or truncated):
            # state shape esperado: (sequence_length, num_features)
            action = agent.select_action(state, add_noise=True)  # (6,)

            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)

            # Guardar transición
            agent.store_transition(state, action, float(reward), next_state, done)

            # Entrenar (si hay suficientes muestras)
            _ = agent.train_step()

            # Acumular métricas
            ep_reward += float(reward)
            ep_pnl += float(step_info.get("pnl_delta_mtm", 0.0)) + float(step_info.get("pnl_settlement", 0.0))
            ep_overhedge += float(step_info.get("sobre_cobertura_kwh", 0.0))
            ep_tx_cost += float(step_info.get("transaction_costs", 0.0))
            ep_ahorro += (float(step_info.get("pnl_delta_mtm", 0.0)) + float(step_info.get("pnl_settlement", 0.0)) - float(step_info.get("transaction_costs", 0.0)))

            # nuevos acumulados para análisis detallado
            ep_coverage_penalty += float(step_info.get("coverage_penalty", 0.0))
            ep_pnl_norm += float(step_info.get("reward_pnl_norm", 0.0))
            ep_risk_norm += float(step_info.get("reward_risk_norm", 0.0))
            ep_overhedge_norm += float(step_info.get("reward_overhedge_norm", 0.0))
            ep_transaction_norm += float(step_info.get("reward_tx_norm", 0.0))
            ep_duplicate_norm += float(step_info.get("reward_duplicate_norm", 0.0))
            ep_opportunity_norm += float(step_info.get("reward_opportunity_norm", 0.0))
            ep_opportunity_expiry_norm += float(step_info.get("reward_opportunity_expiry_norm", 0.0))
            ep_carry_norm += float(step_info.get("reward_carry_norm", 0.0))
            ep_capital_ratio += float(step_info.get("capital_ratio_norm", 0.0))

            # Conteo aproximado de margin calls (si hubo costo > 0 en el step)
            if float(step_info.get("margin_calls_cost", 0.0)) > 0.0:
                ep_margin_calls += 1

            state = next_state
            conteo += 1

        # ------------------------------------------------------------------
        # 5) Decaimiento del ruido
        # ------------------------------------------------------------------
        agent.decay_noise()

        # Tiempo episodio
        ep_time = time.perf_counter() - t0

        # Guardar métricas
        episode_rewards.append(ep_reward)
        episode_pnls.append(ep_pnl)
        margin_calls_count.append(ep_margin_calls)
        overhedging_penalties.append(ep_overhedge)
        episode_times_sec.append(ep_time)
        episode_tx_costs.append(ep_tx_cost)
        episode_ahorros.append(ep_ahorro)
        end_date = step_info.get("current_date", "N/A")

        # ------------------------------------------------------------------
        # 6) Guardado best model
        # ------------------------------------------------------------------
        if ep_ahorro > best_reward:
            best_reward = ep_ahorro
            torch.save(agent.actor.state_dict(), weights_dir / "best_actor_ddpg.pt")
            torch.save(agent.critic.state_dict(), weights_dir / "best_critic_ddpg.pt")
            train_no_improvement_count = 0
            print(f"Nuevo best model guardado con reward {best_reward:,.2f} en episodio {episode}.")
        else:
            train_no_improvement_count += 1


        # Logging en consola
        if episode % log_every == 0 or episode == 1 or episode == total_episodes:
            print(
                f"[Episodio {episode:4d}/{total_episodes}] "
                f"Reward={ep_reward:,.2f} | "
                f"PnL={ep_pnl:,.2f} | "
                f"Ahorro={ep_ahorro:,.2f} | TxCost={ep_tx_cost:,.2f} | "
                f"NoiseStd={agent.noise_std:.4f} | "
                f"Tiempo={ep_time:.2f}s | "
                f"Cap={step_info.get('capital_actual', 0):,.0f} | "
                f"Margin={step_info.get('margin_balance_total', 0):,.0f} | "
                f"PnLNorm={ep_pnl_norm / conteo:,.2f} | "
                f"RiskNorm={ep_risk_norm / conteo:,.2f} | "
                f"OverhedgeNorm={ep_overhedge_norm / conteo:,.2f} | "
                f"TransactionNorm={ep_transaction_norm / conteo:,.2f} | "
                f"OpportunityNorm={ep_opportunity_norm / conteo:,.2f} | "
                f"OpportunityExpiryNorm={ep_opportunity_expiry_norm / conteo:,.2f} | "
                f"CoveragePenalty={ep_coverage_penalty / conteo:,.2f} | "
                f"CapitalRatioNorm={ep_capital_ratio / conteo:,.2f} | "
                f"CarryNorm={ep_carry_norm / conteo:,.2f} | "
                f"Truncated={truncated} | "
                f"Terminated={terminated} | "
                f"EndDate={end_date}"
            )
        
        # Early stopping simple basado en no mejora por N episodios
        if train_no_improvement_count >= train_threshold:
            total_episodes_run = episode
            print(f"Early stopping en episodio {episode} por no mejora en {train_threshold} episodios.")
            break
        else:
            total_episodes_run = total_episodes

    # ------------------------------------------------------------------
    # Guardado de histórico de entrenamiento
    # ------------------------------------------------------------------
    history_df = pd.DataFrame(
        {
            "episode": np.arange(1, total_episodes_run + 1, dtype=int),
            "episode_reward": episode_rewards,
            "episode_pnl": episode_pnls,
            "episode_tx_cost": episode_tx_costs,
            "episode_ahorro_cop": episode_ahorros,
            "margin_calls_count": margin_calls_count,
            "overhedging_penalty_kwh_sum": overhedging_penalties,
            "episode_time_sec": episode_times_sec,
        }
    )
    history_df.to_csv(results_dir / "training_history_ddpg.csv", index=False)

    return {
        "episode_rewards": episode_rewards,
        "episode_pnls": episode_pnls,
        "episode_tx_costs": episode_tx_costs,
        "episode_ahorros": episode_ahorros,
        "margin_calls_count": margin_calls_count,
        "overhedging_penalties": overhedging_penalties,
        "episode_times_sec": episode_times_sec,
    }


#if __name__ == "__main__":
#    train_ddpg_agent(CONFIG)