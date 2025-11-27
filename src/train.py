
from config import Config
from data_loader import load_spot_daily, load_futures_with_calendar, build_calendar_dict
from env_energy_curve_rollover import EnergyCurveRolloverEnv
from replay_buffer import ReplayBuffer
from maddpg_dualq import MADDPG_DualQ
import numpy as np
import torch
import random
import pandas as pd

def main():
    cfg = Config()
    spot_daily = load_spot_daily(cfg)
    df_futuros = load_futures_with_calendar(cfg)
    calendar = build_calendar_dict(df_futuros, cfg)

    env = EnergyCurveRolloverEnv(df_futuros, spot_daily, calendar, cfg)
    n_venc_total = sum(len(calendar[t]) for t in cfg.contract_types)
    act_dim = n_venc_total + len(cfg.contract_types)
    state_dim = 1 + n_venc_total*2

    algo = MADDPG_DualQ(state_dim, act_dim, cfg)
    buf = ReplayBuffer(state_dim, act_dim, cfg.buffer_size)

    s = env.reset()
    for step in range(cfg.train_steps):
        a = algo.act(s, noise_sigma=cfg.act_noise_sigma)
        s2, r_vec, done, info = env.step(a)
        buf.add(s, a, np.mean(r_vec), s2, float(done))
        s = s2
        if buf.size >= cfg.batch_size:
            batch = buf.sample(cfg.batch_size)
            algo.train_step(batch, cfg.gamma, cfg.tau)
        if done:
            s = env.reset()

    print("Entrenamiento finalizado.")