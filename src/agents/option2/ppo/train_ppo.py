"""Script principal de entrenamiento PPO (LSTM) para cobertura ELM."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from config import CONFIG, ProjectConfig
from data_processor import DataProcessor
from environment import ElectricityHedgingEnv
from ppo_agent import PPOAgent
from seed_utils import set_all_seeds


def _resolve_training_hyperparams(config: ProjectConfig) -> Tuple[int, int]:
    """Extrae total_episodes y log_every con fallback seguro."""
    total_episodes = int(getattr(config.general, "total_episodes", 200))
    log_every = int(getattr(config.general, "log_every", 10))
    return total_episodes, log_every


def _resolve_output_dirs(config: ProjectConfig) -> Tuple[Path, Path]:
    """Resuelve directorios para pesos y resultados."""
    if hasattr(config.paths, "model_dir"):
        weights_dir = Path(config.paths.model_dir)
    else:
        weights_dir = Path("src/models/option2/ppo")

    if hasattr(config.paths, "results_dir"):
        results_dir = Path(config.paths.results_dir)
    else:
        results_dir = Path("results/option2/ppo")

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


def train_ppo_agent(config: ProjectConfig = CONFIG) -> Dict[str, List[float]]:
    """Orquesta entrenamiento completo PPO."""

    # ------------------------------------------------------------------
    # 1) Inicialización
    # ------------------------------------------------------------------
    set_all_seeds(config.general.seed, deterministic=True)

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
    agent = PPOAgent(
        num_features=num_features,
        action_dim=config.contract.max_horizon_months,
        config=config,
    )

    total_episodes, log_every = _resolve_training_hyperparams(config)
    weights_dir, results_dir = _resolve_output_dirs(config)

    rollout_steps = int(config.ppo.rollout_steps)
    train_threshold = int(getattr(config.general, "train_threshold", 100))

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

    actor_losses: List[float] = []
    critic_losses: List[float] = []
    entropies: List[float] = []
    approx_kls: List[float] = []

    best_score = -np.inf
    no_improvement_count = 0

    # ------------------------------------------------------------------
    # 3) Bucle de entrenamiento
    # ------------------------------------------------------------------
    for episode in range(1, total_episodes + 1):
        t0 = time.perf_counter()

        ep_seed = int(config.general.seed + episode)
        try:
            state, _ = env.reset(seed=ep_seed)
        except TypeError:
            state = env.reset(seed=ep_seed)

        terminated = False
        truncated = False

        ep_reward = 0.0
        ep_pnl = 0.0
        ep_margin_calls = 0
        ep_overhedge = 0.0
        ep_tx_cost = 0.0
        ep_ahorro = 0.0
        end_date = "N/A"

        steps_in_episode = 0
        updates_in_episode = 0

        while not (terminated or truncated):
            # Selección de acción on-policy
            action, logprob, value = agent.select_action(state, deterministic=False)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            agent.store_transition(
                state=state,
                action=action,
                logprob=logprob,
                reward=float(reward),
                done=done,
                value=value,
            )

            # Métricas episodio
            pnl_step = float(info.get("pnl_delta_mtm", 0.0)) + float(info.get("pnl_settlement", 0.0))
            tx_step = float(info.get("transaction_costs", 0.0))

            ep_reward += float(reward)
            ep_pnl += pnl_step
            ep_tx_cost += tx_step
            ep_ahorro += (pnl_step - tx_step)
            ep_overhedge += float(info.get("sobre_cobertura_kwh", 0.0))
            end_date = str(info.get("current_date", "N/A"))

            if float(info.get("margin_calls_cost", 0.0)) > 0.0:
                ep_margin_calls += 1

            state = next_state
            steps_in_episode += 1

            # Update PPO al llenar rollout o al terminar episodio
            if len(agent.rollout_buffer) >= rollout_steps or done:
                # bootstrap value si no terminó realmente el episodio
                if done:
                    last_value = 0.0
                else:
                    with torch.no_grad():
                        _, _, last_value = agent.select_action(state, deterministic=True)

                upd = agent.update(last_value=last_value)
                actor_losses.append(float(upd.get("actor_loss", 0.0)))
                critic_losses.append(float(upd.get("critic_loss", 0.0)))
                entropies.append(float(upd.get("entropy", 0.0)))
                approx_kls.append(float(upd.get("approx_kl", 0.0)))
                updates_in_episode += 1

        ep_time = time.perf_counter() - t0

        # Guardar métricas
        episode_rewards.append(ep_reward)
        episode_pnls.append(ep_pnl)
        margin_calls_count.append(ep_margin_calls)
        overhedging_penalties.append(ep_overhedge)
        episode_times_sec.append(ep_time)
        episode_tx_costs.append(ep_tx_cost)
        episode_ahorros.append(ep_ahorro)

        # Criterio "best model" (negocio)
        if ep_ahorro > best_score:
            best_score = ep_ahorro
            torch.save(agent.actor.state_dict(), weights_dir / "best_actor_ppo.pt")
            torch.save(agent.critic.state_dict(), weights_dir / "best_critic_ppo.pt")
            no_improvement_count = 0
            print(f"Nuevo best PPO guardado con ahorro {best_score:,.2f} en episodio {episode}.")
        else:
            no_improvement_count += 1

        # Logging
        if episode % log_every == 0 or episode == 1 or episode == total_episodes:
            last_actor_loss = actor_losses[-1] if actor_losses else 0.0
            last_critic_loss = critic_losses[-1] if critic_losses else 0.0
            last_entropy = entropies[-1] if entropies else 0.0
            last_kl = approx_kls[-1] if approx_kls else 0.0

            print(
                f"[Episodio {episode:4d}/{total_episodes}] "
                f"Reward={ep_reward:,.2f} | "
                f"PnL={ep_pnl:,.2f} | "
                f"Ahorro={ep_ahorro:,.2f} | "
                f"TxCost={ep_tx_cost:,.2f} | "
                f"Steps={steps_in_episode} | "
                f"Updates={updates_in_episode} | "
                f"ActorLoss={last_actor_loss:,.4f} | "
                f"CriticLoss={last_critic_loss:,.4f} | "
                f"Entropy={last_entropy:,.4f} | "
                f"KL={last_kl:,.5f} | "
                f"ActionStd={agent.action_std:.4f} | "
                f"Tiempo={ep_time:.2f}s | "
                f"Cap={float(info.get('capital_actual', 0.0)):,.0f} | "
                f"Margin={float(info.get('margin_balance_total', 0.0)):,.0f} | "
                f"Truncated={truncated} | "
                f"Terminated={terminated} | "
                f"EndDate={end_date}"
            )

        # Early stopping
        if no_improvement_count >= train_threshold:
            total_episodes_run = episode
            print(f"Early stopping en episodio {episode} por no mejora en {train_threshold} episodios.")
            break
    else:
        total_episodes_run = total_episodes

    # ------------------------------------------------------------------
    # 4) Guardado de histórico
    # ------------------------------------------------------------------
    history_df = pd.DataFrame(
        {
            "episode": np.arange(1, total_episodes_run + 1, dtype=int),
            "episode_reward": episode_rewards[:total_episodes_run],
            "episode_pnl": episode_pnls[:total_episodes_run],
            "episode_tx_cost": episode_tx_costs[:total_episodes_run],
            "episode_ahorro_cop": episode_ahorros[:total_episodes_run],
            "margin_calls_count": margin_calls_count[:total_episodes_run],
            "overhedging_penalty_kwh_sum": overhedging_penalties[:total_episodes_run],
            "episode_time_sec": episode_times_sec[:total_episodes_run],
            "actor_loss": actor_losses[:total_episodes_run] if len(actor_losses) >= total_episodes_run else [np.nan] * total_episodes_run,
            "critic_loss": critic_losses[:total_episodes_run] if len(critic_losses) >= total_episodes_run else [np.nan] * total_episodes_run,
            "entropy": entropies[:total_episodes_run] if len(entropies) >= total_episodes_run else [np.nan] * total_episodes_run,
            "approx_kl": approx_kls[:total_episodes_run] if len(approx_kls) >= total_episodes_run else [np.nan] * total_episodes_run,
        }
    )
    history_df.to_csv(results_dir / "training_history_ppo.csv", index=False)

    return {
        "episode_rewards": episode_rewards,
        "episode_pnls": episode_pnls,
        "episode_tx_costs": episode_tx_costs,
        "episode_ahorros": episode_ahorros,
        "margin_calls_count": margin_calls_count,
        "overhedging_penalties": overhedging_penalties,
        "episode_times_sec": episode_times_sec,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "entropies": entropies,
        "approx_kls": approx_kls,
    }


# if __name__ == "__main__":
#     train_ppo_agent(CONFIG)