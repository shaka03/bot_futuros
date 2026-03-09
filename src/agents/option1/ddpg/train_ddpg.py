"""Script principal de entrenamiento DDPG (Fase 5) con split temporal train/test."""

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
    """Extrae total_episodes y log_every con fallback."""
    total_episodes = int(getattr(config.general, "total_episodes", 200))
    log_every = int(getattr(config.general, "log_every", 10))
    return total_episodes, log_every


def _resolve_output_dirs(config: ProjectConfig) -> Tuple[Path, Path]:
    """Resuelve directorios para pesos y resultados."""
    weights_dir = Path(getattr(config.paths, "agents_output_dir", "src/models/option1/ddpg"))
    results_dir = Path(getattr(config.paths, "results_dir", "results/option1"))
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir, results_dir


def _split_train_bundle(bundle, processor: DataProcessor, config: ProjectConfig):
    """Construye partición temporal de entrenamiento: (1 - test_ratio)."""
    test_ratio = float(getattr(config.general, "test_ratio", 0.1))
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"config.general.test_ratio inválido: {test_ratio}. Debe estar entre (0,1).")

    total_seq = int(bundle.lstm_sequences.shape[0])
    cut = max(1, int(total_seq * (1.0 - test_ratio)))

    seq_train = bundle.lstm_sequences[:cut].copy()

    full_timeline = bundle.nemotecnico_map_t1_t6.index
    seq_len = int(config.lstm.sequence_length)

    t_start = seq_len - 1
    t_end_exclusive = min(len(full_timeline), cut + seq_len - 1)
    train_timeline = full_timeline[t_start:t_end_exclusive]

    if len(train_timeline) == 0:
        raise ValueError("train_timeline quedó vacío. Revisa sequence_length y tamaño del dataset.")

    nem_train = bundle.nemotecnico_map_t1_t6.loc[train_timeline].copy()
    dem_train = bundle.demand_aligned.loc[train_timeline].copy()

    fut = bundle.futures_lookup.reset_index()
    fut = fut[fut["Fecha"].isin(train_timeline)].copy()
    fut_train = fut.set_index(["Fecha", "Nemotecnico"]).sort_index()

    liq_df = (
        processor.precios_liquidacion_df.copy()
        if getattr(processor, "precios_liquidacion_df", None) is not None
        else pd.DataFrame()
    )

    return seq_train, fut_train, nem_train, dem_train, liq_df


def train_ddpg_agent(config: ProjectConfig = CONFIG) -> Dict[str, List[float]]:
    """Orquesta entrenamiento completo DDPG sobre partición train."""
    processor = DataProcessor(config)
    bundle = processor.get_agent_data("ELM")

    seq_train, fut_train, nem_train, dem_train, liq_train = _split_train_bundle(bundle, processor, config)

    env = ElectricityHedgingEnv(
        sequences_lstm=seq_train,
        futures_lookup=fut_train,
        nemotecnico_map_t1_t6=nem_train,
        demand_aligned=dem_train,
        precios_liquidacion=liq_train,
        initial_capital=bundle.dynamic_initial_capital,
        config=config,
    )

    num_features = int(seq_train.shape[2])
    agent = DDPGAgent(
        num_features=num_features,
        action_dim=int(config.contract.max_horizon_months),
        config=config,
    )

    total_episodes, log_every = _resolve_training_hyperparams(config)
    weights_dir, results_dir = _resolve_output_dirs(config)

    # Métricas
    episode_rewards: List[float] = []
    episode_pnls: List[float] = []
    margin_calls_count: List[int] = []
    overhedging_penalties: List[float] = []
    episode_times_sec: List[float] = []

    # Nuevas métricas para score combinado
    episode_scores: List[float] = []
    reward_norm_hist: List[float] = []
    pnl_norm_hist: List[float] = []

    best_score = -np.inf

    # EMA para normalización robusta de escalas
    ema_abs_reward = 1.0
    ema_abs_pnl = 1.0
    alpha = 0.05  # suavizado EMA

    # Pesos del score combinado (ajustables)
    w_reward = float(getattr(config.general, "best_model_weight_reward", 0.5))
    w_pnl = float(getattr(config.general, "best_model_weight_pnl", 0.5))

    for episode in range(1, total_episodes + 1):
        t0 = time.perf_counter()

        state, _ = env.reset()
        terminated = False
        truncated = False

        ep_reward = 0.0
        ep_pnl = 0.0
        ep_margin_calls = 0
        ep_overhedge = 0.0

        while not (terminated or truncated):
            action = agent.select_action(state, add_noise=True)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            agent.store_transition(state, action, float(reward), next_state, done)
            _ = agent.train_step()

            ep_reward += float(reward)
            ep_pnl += float(info.get("pnl_delta_mtm", 0.0)) + float(info.get("pnl_settlement", 0.0))
            ep_overhedge += float(info.get("sobre_cobertura_kwh", 0.0))

            if float(info.get("margin_calls_cost", 0.0)) > 0.0:
                ep_margin_calls += 1

            state = next_state

        agent.decay_noise()
        ep_time = time.perf_counter() - t0

        # actualizar EMAs
        ema_abs_reward = (1 - alpha) * ema_abs_reward + alpha * max(1.0, abs(ep_reward))
        ema_abs_pnl = (1 - alpha) * ema_abs_pnl + alpha * max(1.0, abs(ep_pnl))

        reward_norm = ep_reward / ema_abs_reward
        pnl_norm = ep_pnl / ema_abs_pnl
        episode_score = (w_reward * reward_norm) + (w_pnl * pnl_norm)

        # guardar métricas
        episode_rewards.append(ep_reward)
        episode_pnls.append(ep_pnl)
        margin_calls_count.append(ep_margin_calls)
        overhedging_penalties.append(ep_overhedge)
        episode_times_sec.append(ep_time)

        episode_scores.append(float(episode_score))
        reward_norm_hist.append(float(reward_norm))
        pnl_norm_hist.append(float(pnl_norm))

        # Guardar mejor modelo por score combinado
        if episode_score > best_score:
            best_score = episode_score
            torch.save(agent.actor.state_dict(), weights_dir / "best_actor_ddpg.pt")
            torch.save(agent.critic.state_dict(), weights_dir / "best_critic_ddpg.pt")

        if episode % log_every == 0 or episode == 1 or episode == total_episodes:
            print(
                f"[Episodio {episode:4d}/{total_episodes}] "
                f"Reward={ep_reward:,.2f} | "
                f"PnL={ep_pnl:,.2f} | "
                f"Score={episode_score:.4f} | "
                f"NoiseStd={agent.noise_std:.4f} | "
                f"Tiempo={ep_time:.2f}s"
            )

    # Export históricos
    history_df = pd.DataFrame(
        {
            "episode": np.arange(1, total_episodes + 1, dtype=int),
            "episode_reward": episode_rewards,
            "episode_pnl": episode_pnls,
            "reward_norm": reward_norm_hist,
            "pnl_norm": pnl_norm_hist,
            "episode_score": episode_scores,
            "margin_calls_count": margin_calls_count,
            "overhedging_penalty_kwh_sum": overhedging_penalties,
            "episode_time_sec": episode_times_sec,
        }
    )
    history_df.to_csv(results_dir / "training_history_ddpg.csv", index=False)

    return {
        "episode_rewards": episode_rewards,
        "episode_pnls": episode_pnls,
        "episode_scores": episode_scores,
        "margin_calls_count": margin_calls_count,
        "overhedging_penalties": overhedging_penalties,
        "episode_times_sec": episode_times_sec,
    }


# if __name__ == "__main__":
#     train_ddpg_agent(CONFIG)