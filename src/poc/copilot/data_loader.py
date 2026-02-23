
import pandas as pd
from config import Config

def load_spot_daily(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.path_spot)
    df = df[(df[cfg.energia_var_col] == cfg.spot_variable) &
            (df[cfg.energia_duracion_col] == "PT1H")]
    df["Fecha"] = pd.to_datetime(df[cfg.energia_fecha_col]).dt.date
    spot_daily = df.groupby("Fecha")[cfg.energia_valor_col].mean().rename("Spot").to_frame()
    spot_daily.index = pd.to_datetime(spot_daily.index)
    return spot_daily

def load_futures_with_calendar(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.path_futuros)
    df[cfg.date_col_futuros] = pd.to_datetime(df[cfg.date_col_futuros])
    df[cfg.precio_col] = pd.to_numeric(df[cfg.precio_col], errors="coerce")
    df = df[df[cfg.tipo_col].isin(cfg.contract_types)]
    df["Vencimiento"] = pd.to_datetime(df[cfg.anio_col].astype(str) + "-" +
                                       df[cfg.mes_col].astype(str) + "-01") + pd.offsets.MonthEnd(0)
    return df

def build_calendar_dict(df: pd.DataFrame, cfg: Config):
    calendar = {tipo: [] for tipo in cfg.contract_types}
    for tipo in cfg.contract_types:
        sub = df[df[cfg.tipo_col] == tipo].sort_values(cfg.date_col_futuros)
        for venc in sub["Vencimiento"].unique():
            inicio = sub[sub["Vencimiento"] == venc][cfg.date_col_futuros].min()
            nemotecnico = sub[sub["Vencimiento"] == venc]["Nemotecnico"].iloc[0]
            calendar[tipo].append((inicio, venc, nemotecnico))
    return calendar