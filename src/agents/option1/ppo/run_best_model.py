from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

from config import CONFIG
from train_ppo import train_ppo_agent
from evaluate_and_plot import (
    evaluate_agent_out_of_sample,
    build_financial_report,
    evaluate_and_plot,
)


def apply_hp(cfg, hp):
    """Aplica hiperparámetros (reward/lstm/ppo/general) sobre una copia del config."""
    cfg = copy.deepcopy(cfg)

    cfg = replace(
        cfg,
        reward=replace(cfg.reward, **hp["reward"]),
        lstm=replace(cfg.lstm, **hp["lstm"]),
        ppo=replace(cfg.ppo, **hp["ppo"]),
        general=replace(cfg.general, **hp["general"]),
    )
    return cfg


def main() -> None:
    # Ruta por defecto generada por tune_ppo.py
    best_path = Path(CONFIG.paths.results_dir) / "hparam_search" / "best_trial_summary_ppo.json"
    if not best_path.exists():
        raise FileNotFoundError(f"No se encontró resumen del mejor trial: {best_path}")

    best = json.loads(best_path.read_text(encoding="utf-8"))
    hp = json.loads(best["hp_json"])

    cfg = apply_hp(CONFIG, hp)

    # Reutiliza semilla del mejor trial (si existe)
    if "seed" in best:
        cfg = replace(cfg, general=replace(cfg.general, seed=int(best["seed"])))

    # 1) Re-entrenar con best hp (PPO)
    train_ppo_agent(cfg)

    # 2) Evaluar OOS + reporte financiero
    outs, report_df, monthly_df = evaluate_and_plot(cfg)

    # 3) Carpeta dedicada de corrida final
    out_dir = Path(cfg.paths.results_dir) / "best_model_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    outs["eval_df"].to_csv(out_dir / "evaluation_rollout_ppo.csv", index=False)
    report_df.to_csv(out_dir / "financial_report_ppo.csv", index=False)
    monthly_df.to_csv(out_dir / "monthly_comparison_ppo.csv", index=False)

    print("Listo. Resultados en:", out_dir)


# if __name__ == "__main__":
#     main()