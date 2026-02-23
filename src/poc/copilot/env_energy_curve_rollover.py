
import numpy as np
import pandas as pd
from typing import Dict
from config import Config

class EnergyCurveRolloverEnv:
    def __init__(self, df_futuros: pd.DataFrame, spot_daily: pd.DataFrame, calendar: Dict, cfg: Config):
        self.cfg = cfg
        self.df_futuros = df_futuros
        self.spot_daily = spot_daily
        self.calendar = calendar
        self.contract_types = cfg.contract_types
        self.positions = {tipo: np.zeros(len(calendar[tipo]), dtype=np.float32) for tipo in self.contract_types}
        self.reset()

    def reset(self):
        self.t = 1
        for tipo in self.contract_types:
            self.positions[tipo].fill(0.0)
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        date_prev = self.spot_daily.index[self.t-1]
        spot_prev = self.spot_daily.iloc[self.t-1]["Spot"]
        obs_prices, obs_positions = [], []
        for tipo in self.contract_types:
            for m, (_, venc, _) in enumerate(self.calendar[tipo]):
                sub = self.df_futuros[(self.df_futuros[self.cfg.tipo_col] == tipo) &
                                      (self.df_futuros["Vencimiento"] == venc)]
                price_prev = sub[sub[self.cfg.date_col_futuros] <= date_prev][self.cfg.precio_col].last()
                obs_prices.append(price_prev if price_prev is not None else np.nan)
                obs_positions.append(self.positions[tipo][m])
        return np.array([spot_prev] + obs_prices + obs_positions, dtype=np.float32)

    def step(self, actions: np.ndarray):
        n_venc_total = sum(len(self.calendar[t]) for t in self.contract_types)
        pos_actions = actions[:n_venc_total]
        roll_signals = actions[n_venc_total:]

        idx = 0
        for tipo in self.contract_types:
            for m in range(len(self.calendar[tipo])):
                delta = np.clip(pos_actions[idx], -1.0, 1.0) * self.cfg.volume_scale
                self.positions[tipo][m] = np.clip(self.positions[tipo][m] + delta,
                                                  -self.cfg.position_limit, self.cfg.position_limit)
                idx += 1

        date_t = self.spot_daily.index[self.t]
        pnl_fut, rollover_cost = 0.0, 0.0

        for tipo_i, tipo in enumerate(self.contract_types):
            for m, (_, venc, _) in enumerate(self.calendar[tipo]):
                sub = self.df_futuros[(self.df_futuros[self.cfg.tipo_col] == tipo) &
                                      (self.df_futuros["Vencimiento"] == venc)]
                price_t = sub[sub[self.cfg.date_col_futuros] <= date_t][self.cfg.precio_col].last()
                price_tm1 = sub[sub[self.cfg.date_col_futuros] <= self.spot_daily.index[self.t-1]][self.cfg.precio_col].last()
                if price_t is None or price_tm1 is None:
                    continue
                pnl_fut += self.positions[tipo][m] * (price_t - price_tm1)

                if date_t > venc and self.positions[tipo][m] != 0:
                    if roll_signals[tipo_i] >= 0.5 and m+1 < len(self.calendar[tipo]):
                        next_venc = self.calendar[tipo][m+1][1]
                        sub_next = self.df_futuros[(self.df_futuros[self.cfg.tipo_col] == tipo) &
                                                   (self.df_futuros["Vencimiento"] == next_venc)]
                        next_price = sub_next[sub_next[self.cfg.date_col_futuros] <= date_t][self.cfg.precio_col].last()
                        if next_price is not None:
                            rollover_cost += abs(self.positions[tipo][m]) * (next_price - price_t) + self.cfg.rollover_commission
                            self.positions[tipo][m+1] += self.positions[tipo][m]
                            self.positions[tipo][m] = 0.0
                    else:
                        rollover_cost += self.cfg.close_commission
                        self.positions[tipo][m] = 0.0

        cost_spot = self.spot_daily.iloc[self.t]["Spot"] * self.cfg.hedge_notional
        cost_net = cost_spot - pnl_fut + rollover_cost

        trade_penalty = self.cfg.tc_lambda * np.sum(np.abs(pos_actions))
        reward = -(cost_net**2) - trade_penalty - self.cfg.rollover_penalty * rollover_cost
        rewards = np.full(len(actions), reward, dtype=np.float32)

        self.t += 1
        if self.t >= len(self.spot_daily)-1:
            self.done = True

        info = {"cost_net": cost_net, "pnl_fut": pnl_fut, "rollover_cost": rollover_cost}
        return self._get_obs(), rewards, self.done, info