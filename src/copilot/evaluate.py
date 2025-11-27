
from config import Config
from data_loader import load_spot_daily, load_futures_with_calendar, build_calendar_dict
from env_energy_curve_rollover import EnergyCurveRolloverEnv
from maddpg_dualq import MADDPG_DualQ
import matplotlib.pyplot as plt

def main():
    cfg = Config()
    spot_daily = load_spot_daily(cfg)
    df_futuros = load_futures_with_calendar(cfg)
    calendar = build_calendar_dict(df_futuros, cfg)
    env = EnergyCurveRolloverEnv(df_futuros, spot_daily, calendar, cfg)

    n_venc_total = sum(len(calendar[t]) for t in cfg.contract_types)
    act_dim = n_venc_total + len(cfg.contract_types)
    state_dim = 1 + n_venc_total*2

    policy = MADDPG_DualQ(state_dim, act_dim, cfg)  # cargar pesos si los guardaste
    s = env.reset()
    costs, roll_costs = [], []
    while True:
        a = policy.act(s, noise_sigma=0.0)
        s, r_vec, done, info = env.step(a)
        costs.append(info["cost_net"])
        roll_costs.append(info["rollover_cost"])
        if done: break

    plt.figure(figsize=(12,6))
    plt.plot(costs, label="Costo neto")
    plt.plot(roll_costs, label="Costo roll-over")
    plt.legend()
    plt.title("Evaluación con curva completa y roll-over")
    plt.savefig("eval_curve_rollover.png")
    print("Gráfico guardado en eval_curve_rollover.png")

if __name__ == "__main__":
    main()