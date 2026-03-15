"""Random Search de hiperparámetros para PPO (LSTM).

Uso:
    python tune_ppo.py --trials 20 --mode train_eval

Notas:
- Asume que ya tienes:
    - config.py (ProjectConfig/CONFIG con sección ppo)
    - train_ppo.py (train_ppo_agent)
    - evaluate_and_plot.py (evaluate_agent_out_of_sample, build_financial_report)
- Guarda resultados en results/option1/ppo/hparam_search
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from config import CONFIG, ProjectConfig
from train_ppo import train_ppo_agent
from evaluate_and_plot import evaluate_agent_out_of_sample, build_financial_report
import run_best_model


# -----------------------------
# Utilidades
# -----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patch_config(base: ProjectConfig, hp: Dict[str, Any]) -> ProjectConfig:
    """Copia config base y aplica hiperparámetros en reward/lstm/ppo/general."""
    cfg = copy.deepcopy(base)

    cfg = replace(
        cfg,
        reward=replace(cfg.reward, **hp["reward"]),
        lstm=replace(cfg.lstm, **hp["lstm"]),
        ppo=replace(cfg.ppo, **hp["ppo"]),
        general=replace(cfg.general, **hp["general"]),
    )

    return cfg


def sample_hyperparams(rng: random.Random) -> Dict[str, Dict[str, Any]]:
    """Muestreo aleatorio (random search) de hiperparámetros PPO."""
    reward = {
        # pesos
        "w_pnl": rng.choice([0.20, 0.30, 0.40, 0.50, 0.60]),
        "w_coverage": rng.choice([0.10, 0.20, 0.25, 0.35, 0.50]),
        "w_transaction": rng.choice([0.05, 0.10, 0.15, 0.20]),
        "w_opportunity": rng.choice([0.00, 0.10, 0.20]),
        "w_opportunity_expiry": rng.choice([0.00, 0.05, 0.10, 0.20]),
        "w_capital_stress": rng.choice([0.40, 0.60, 0.80, 1.00, 1.20]),
        "w_margin_call": rng.choice([0.05, 0.10, 0.20, 0.30, 0.50]),
        "w_carry": rng.choice([0.00, 0.05, 0.08, 0.12, 0.15]),
        "w_risk": rng.choice([0.00, 0.05, 0.08, 0.12]),
        "w_overhedge": rng.choice([0.00, 0.05, 0.08, 0.12]),
        # escalas (fijas base)
        "scale_pnl": 5e7,
        "scale_money": 1e7,
        "scale_tx": 1e6,
        "scale_kwh": 5e6,
        "scale_opportunity": 1e9,
        "scale_opportunity_expiry": 8e5,
        "scale_risk": 8e15,
        "scale_carry": 1e8,
    }

    lstm = {
        "sequence_length": rng.choice([14, 21, 30]),
        "hidden_size": rng.choice([64, 128, 256]),
        "num_layers": rng.choice([1, 2]),
        "dropout": rng.choice([0.0, 0.1]),
    }

    ppo = {
        "actor_lr": rng.choice([1e-5, 3e-5, 1e-4]),
        "critic_lr": rng.choice([5e-5, 1e-4, 3e-4]),
        "gamma": rng.choice([0.97, 0.99]),
        "gae_lambda": rng.choice([0.90, 0.95, 0.97]),
        "clip_eps": rng.choice([0.10, 0.15, 0.20, 0.25]),
        "entropy_coef": rng.choice([0.0, 0.005, 0.01, 0.02]),
        "value_coef": rng.choice([0.25, 0.50, 0.75]),
        "max_grad_norm": rng.choice([0.3, 0.5, 1.0]),
        "target_kl": rng.choice([0.01, 0.02, 0.03, 0.05]),
        "rollout_steps": rng.choice([512, 1024, 2048]),
        "ppo_epochs": rng.choice([5, 10, 15]),
        "mini_batch_size": rng.choice([128, 256, 512]),
        "action_std_init": rng.choice([0.10, 0.15, 0.20, 0.25]),
        "action_std_min": rng.choice([0.03, 0.05, 0.08]),
        "action_std_decay": rng.choice([0.9990, 0.9995, 0.9998]),
    }

    general = {
        "discretize_limit": rng.choice([0.35, 0.40, 0.45, 0.50]),
        "total_episodes": rng.choice([100, 150, 200]),
        "train_threshold": 100,
        "test_ratio": 0.09,
    }

    return {"reward": reward, "lstm": lstm, "ppo": ppo, "general": general}


def compute_objective(report_row: pd.Series) -> float:
    """
    Objetivo compuesto (maximizar):
      + ahorro
      - penalización por margin calls en COP
      - penalización por volatilidad estrategia
    """
    ahorro = float(report_row["Ahorro_Total_COP"])
    # Puedes activar penalizaciones si quieres:
    # mc_cost = float(report_row["Total_Margin_Calls_COP"])
    # vol_strat = float(report_row["Volatilidad_Estrategia_std"])
    # score = ahorro - 0.25 * mc_cost - 5.0 * vol_strat
    score = ahorro
    return float(score)


# -----------------------------
# Loop de búsqueda
# -----------------------------
def run_random_search(
    base_config: ProjectConfig,
    trials: int,
    seed: int,
    mode: str = "train_eval",
    run_best_after_search: bool = False,
) -> pd.DataFrame:
    """
    mode:
      - train_only: solo entrena y guarda métricas train
      - train_eval: entrena + eval OOS + reporte financiero
    """
    set_global_seed(seed)
    rng = random.Random(seed)

    out_dir = Path(base_config.paths.results_dir) / "hparam_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for t in range(1, trials + 1):
        hp = sample_hyperparams(rng)
        cfg = patch_config(base_config, hp)

        trial_seed = seed + t
        cfg = replace(cfg, general=replace(cfg.general, seed=trial_seed))
        set_global_seed(trial_seed)

        print(f"\n=== Trial {t}/{trials} | seed={trial_seed} ===")
        print(json.dumps(hp, indent=2))

        # 1) Train
        train_out = train_ppo_agent(cfg)
        train_pnl_last = float(np.mean(train_out["episode_pnls"][-10:])) if train_out["episode_pnls"] else 0.0
        train_reward_last = float(np.mean(train_out["episode_rewards"][-10:])) if train_out["episode_rewards"] else 0.0

        row: Dict[str, Any] = {
            "trial": t,
            "seed": trial_seed,
            "train_reward_mean": train_reward_last,
            "train_pnl_mean": train_pnl_last,
            "hp_json": json.dumps(hp),
        }

        # 2) Eval (opcional)
        if mode == "train_eval":
            outs = evaluate_agent_out_of_sample(cfg)
            eval_df = outs["eval_df"]
            spot_daily = outs["spot_daily"]
            report_df, _ = build_financial_report(eval_df, spot_daily)
            rep = report_df.iloc[0]

            score = compute_objective(rep)

            row.update(
                {
                    "score": score,
                    "Ahorro_Total_COP": float(rep["Ahorro_Total_COP"]),
                    "Costo_Total_Spot_COP": float(rep["Costo_Total_Spot_COP"]),
                    "Costo_Total_Estrategia_PPO_COP": float(rep.get("Costo_Total_Estrategia_PPO_COP", np.nan)),
                    "Reduccion_Volatilidad_COP_std": float(rep["Reduccion_Volatilidad_COP_std"]),
                    "Volatilidad_Estrategia_std": float(rep["Volatilidad_Estrategia_std"]),
                    "Total_Margin_Calls": int(rep["Total_Margin_Calls"]),
                    "Total_Margin_Calls_COP": float(rep["Total_Margin_Calls_COP"]),
                    "Sharpe_PnL": float(rep["Sharpe_PnL"]),
                }
            )

            print(
                f"Trial {t} score={score:,.2f} | "
                f"ahorro={row['Ahorro_Total_COP']:,.2f} | "
                f"mc={row['Total_Margin_Calls_COP']:,.2f}"
            )

        rows.append(row)

        # Guardado incremental
        df_tmp = pd.DataFrame(rows)
        df_tmp.to_csv(out_dir / "random_search_results_ppo.csv", index=False)

    df = pd.DataFrame(rows)

    # Ranking
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values("train_pnl_mean", ascending=False).reset_index(drop=True)

    df.to_csv(out_dir / "random_search_results_ppo.csv", index=False)

    # Guardar top-1
    top = df.iloc[0].to_dict()
    with open(out_dir / "best_trial_summary_ppo.json", "w", encoding="utf-8") as f:
        json.dump(top, f, ensure_ascii=False, indent=2)

    print("\n=== BEST TRIAL PPO ===")
    print(json.dumps(top, indent=2, ensure_ascii=False))
    print(f"\nResultados en: {out_dir}")

    # Ejecutar mejor modelo al finalizar
    run_best_model.main()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Search PPO")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--mode", type=str, choices=["train_only", "train_eval"], default="train_eval")
    args = parser.parse_args()

    run_random_search(
        base_config=CONFIG,
        trials=args.trials,
        seed=args.seed,
        mode=args.mode,
        run_best_after_search=args.run_best_after_search,
    )


if __name__ == "__main__":
    main()