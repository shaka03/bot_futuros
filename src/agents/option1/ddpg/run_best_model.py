from __future__ import annotations
import json
from pathlib import Path
import copy

from config import CONFIG
from train_ddpg import train_ddpg_agent
from evaluate_and_plot import evaluate_agent_out_of_sample, build_financial_report

def apply_hp(cfg, hp):
    cfg = copy.deepcopy(cfg)
    for sec in ["reward", "lstm", "ddpg", "general"]:
        for k, v in hp[sec].items():
            setattr(getattr(cfg, sec), k, v)
    return cfg

def main():
    best_path = Path(CONFIG.paths.results_dir) / "hparam_search/option1/ddpg/" / "best_trial_summary.json"
    best = json.loads(best_path.read_text(encoding="utf-8"))
    hp = json.loads(best["hp_json"])

    cfg = apply_hp(CONFIG, hp)
    cfg.general.seed = int(best["seed"])  # opcional: misma seed del best

    # 1) Re-entrenar con best hp
    train_ddpg_agent(cfg)

    # 2) Evaluar OOS y reporte financiero
    outs = evaluate_agent_out_of_sample(cfg)
    report_df, monthly_df = build_financial_report(outs["eval_df"], outs["spot_daily"])

    out_dir = Path(cfg.paths.results_dir) / "best_model_run/option1/ddpg/"
    out_dir.mkdir(parents=True, exist_ok=True)
    outs["eval_df"].to_csv(out_dir / "evaluation_rollout_ddpg.csv", index=False)
    report_df.to_csv(out_dir / "financial_report_ddpg.csv", index=False)
    monthly_df.to_csv(out_dir / "monthly_comparison_ddpg.csv", index=False)

    print("Listo. Resultados en:", out_dir)

if __name__ == "__main__":
    main()