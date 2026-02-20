import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent  # папка, где лежит mgmo.py
SALES_PATH = BASE_DIR / "sales_train_validation.csv"
CAL_PATH   = BASE_DIR / "calendar.csv"

H = 28
MAX_LAG = 28


def wape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum(np.abs(y_true))
    return np.sum(np.abs(y_true - y_pred)) / denom if denom != 0 else np.nan


def load_aggregate_series(level="state", key="CA"):
    sales = pd.read_csv(SALES_PATH)
    cal = pd.read_csv(CAL_PATH)

    day_cols = [c for c in sales.columns if c.startswith("d_")]

    if level == "state":
        s = sales[sales["state_id"] == key]
        snap_col = f"snap_{key}"
    elif level == "store":
        s = sales[sales["store_id"] == key]
        # store_id вида "CA_1" -> штат "CA"
        snap_col = f"snap_{key.split('_')[0]}"
    else:
        raise ValueError("level must be 'state' or 'store'")

    # 1 временной ряд: сумма продаж по выбранному региону/магазину
    y = s[day_cols].sum(axis=0).rename("sales").reset_index()
    y.columns = ["d", "sales"]

    # календарь
    keep = ["d", "date", "wday", "month", "year", "wm_yr_wk", "event_name_1", "event_name_2", snap_col]
    cal2 = cal[keep].copy()
    y = y.merge(cal2, on="d", how="left")

    y["date"] = pd.to_datetime(y["date"])
    y = y.sort_values("date").reset_index(drop=True)

    # is_event = есть ли событие/праздник
    y["is_event"] = ((~y["event_name_1"].isna()) | (~y["event_name_2"].isna())).astype(int)

    # переименуем snap в единое имя
    y.rename(columns={snap_col: "snap"}, inplace=True)

    return y


def add_lag_features(df):
    df = df.copy()
    df["t"] = np.arange(len(df))  # индекс времени

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    for win in [7, 28]:
        df[f"roll_mean_{win}"] = df["sales"].shift(1).rolling(win).mean()
        df[f"roll_std_{win}"]  = df["sales"].shift(1).rolling(win).std()

    return df


def make_one_model_with_offset_dataset(df, train_end_idx, H=28, max_lag=28, val_days=56):
    """
    Строим обучающую таблицу:
    базовый день t + горизонт h -> таргет sales[t+h]
    """
    df = df.copy()

    base_feat_cols = [
        "t", "wday", "month", "year", "wm_yr_wk", "snap", "is_event",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_28", "roll_std_7", "roll_std_28"
    ]
    fut_feat_cols = ["wday", "month", "year", "wm_yr_wk", "snap", "is_event"]

    # base t так, чтобы t+h не залезал в тест (target <= train_end_idx)
    base_indices = np.arange(max_lag, train_end_idx - H + 1)
    h_vals = np.arange(1, H + 1)

    base_rep = np.repeat(base_indices, H)
    h_rep = np.tile(h_vals, len(base_indices))
    target_idx = base_rep + h_rep

    X_base = df.loc[base_rep, base_feat_cols].reset_index(drop=True).add_prefix("base_")
    X_fut  = df.loc[target_idx, fut_feat_cols].reset_index(drop=True).add_prefix("fut_")

    X = pd.concat([X_base, X_fut], axis=1)
    X["h"] = h_rep

    y = df.loc[target_idx, "sales"].to_numpy()
    target_dates = df.loc[target_idx, "date"].reset_index(drop=True)

    # простая time-based валидация внутри train для early stopping
    train_end_date = df.loc[train_end_idx, "date"]
    val_start_date = train_end_date - pd.Timedelta(days=val_days - 1)

    train_mask = target_dates < val_start_date
    val_mask = ~train_mask

    return X, y, train_mask.to_numpy(), val_mask.to_numpy()


def train_models(X, y, train_mask, val_mask):
    X_train, y_train = X.loc[train_mask], y[train_mask]
    X_val, y_val     = X.loc[val_mask], y[val_mask]

    xgb_model = xgb.XGBRegressor(
        n_estimators=4000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        eval_metric="mae",
        early_stopping_rounds=150,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=8000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="regression",
        random_state=42,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(300, verbose=False)]
    )

    return xgb_model, lgb_model


def one_shot_forecast(df, model, train_end_idx, H=28):
    base_feat_cols = [
        "t", "wday", "month", "year", "wm_yr_wk", "snap", "is_event",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_28", "roll_std_7", "roll_std_28"
    ]
    fut_feat_cols = ["wday", "month", "year", "wm_yr_wk", "snap", "is_event"]

    base_i = train_end_idx
    h = np.arange(1, H + 1)
    base_rep = np.repeat(base_i, H)
    target_idx = base_rep + h

    X_base = df.loc[base_rep, base_feat_cols].reset_index(drop=True).add_prefix("base_")
    X_fut  = df.loc[target_idx, fut_feat_cols].reset_index(drop=True).add_prefix("fut_")

    X_test = pd.concat([X_base, X_fut], axis=1)
    X_test["h"] = h

    y_true = df.loc[target_idx, "sales"].to_numpy()
    dates  = df.loc[target_idx, "date"].to_numpy()

    y_pred = model.predict(X_test)
    return dates, y_true, y_pred


# ====== RUN ======
# 1) агрегируем
df = load_aggregate_series(level="state", key="CA")  # <-- поменяй на level="store", key="CA_1" если хочешь магазин
df = add_lag_features(df)

# 2) holdout: последние 28 дней
train_end_idx = len(df) - H - 1

# 3) обучающая таблица one-model-with-offset
X, y, train_mask, val_mask = make_one_model_with_offset_dataset(df, train_end_idx, H=H, max_lag=MAX_LAG)

# 4) обучение
xgb_model, lgb_model = train_models(X, y, train_mask, val_mask)

# 5) one-shot прогноз + WAPE
dates, y_true, y_pred_xgb = one_shot_forecast(df, xgb_model, train_end_idx, H=H)
_,     _,     y_pred_lgb  = one_shot_forecast(df, lgb_model, train_end_idx, H=H)

print("WAPE XGBoost:", wape(y_true, y_pred_xgb))
print("WAPE LightGBM:", wape(y_true, y_pred_lgb))

# график
plt.figure(figsize=(10,4))
plt.plot(dates, y_true, label="Actual")
plt.plot(dates, y_pred_xgb, label="XGBoost")
plt.plot(dates, y_pred_lgb, label="LightGBM")
plt.xticks(rotation=45)
plt.title("One-shot forecast (28 days)")
plt.legend()
plt.tight_layout()
plt.show()