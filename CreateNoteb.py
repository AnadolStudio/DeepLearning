# Create a well-structured Jupyter Notebook for the course project

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

notebook = new_notebook()

# =========================
# Markdown Cells
# =========================

notebook.cells.append(new_markdown_cell(
"# Курсовой проект\n"
"## Разработка системы прогнозирования спроса на товары\n\n"
"**Датасет:** M5 Forecasting (Walmart)\n\n"
"**Уровень агрегации:** Суммарный спрос по штату (CA)\n\n"
"**Модели:** XGBoost и LightGBM\n\n"
"**Горизонт прогноза:** 28 дней\n\n"
"**Метрика:** WAPE (Weighted Absolute Percentage Error)"
))

notebook.cells.append(new_markdown_cell(
"## 1. Постановка задачи\n\n"
"Цель проекта — разработать систему прогнозирования спроса на товары "
"на основе исторических данных продаж.\n\n"
"В рамках данной работы:\n"
"- используется агрегированный временной ряд продаж по штату CA;\n"
"- выполняется прогноз на 28 дней вперёд;\n"
"- проводится сравнение моделей XGBoost и LightGBM;\n"
"- качество оценивается с помощью метрики WAPE."
))

notebook.cells.append(new_markdown_cell(
"## 2. Загрузка данных\n\n"
"Используются два файла датасета M5:\n"
"- `sales_train_validation.csv`\n"
"- `calendar.csv`\n\n"
"Данные агрегируются по штату CA."
))

# =========================
# Code Cells
# =========================

notebook.cells.append(new_code_cell(
"import pandas as pd\n"
"import numpy as np\n"
"import xgboost as xgb\n"
"import lightgbm as lgb\n"
"import matplotlib.pyplot as plt\n\n"
"SALES_PATH = 'sales_train_validation.csv'\n"
"CAL_PATH = 'calendar.csv'\n"
"H = 28\n"
"MAX_LAG = 28\n"
"\n"
"def wape(y_true, y_pred):\n"
"    y_true = np.asarray(y_true)\n"
"    y_pred = np.asarray(y_pred)\n"
"    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))"
))

notebook.cells.append(new_markdown_cell(
"## 3. Подготовка данных\n\n"
"Агрегируем продажи по штату CA и объединяем с календарными признаками."
))

notebook.cells.append(new_code_cell(
"sales = pd.read_csv(SALES_PATH)\n"
"calendar = pd.read_csv(CAL_PATH)\n\n"
"day_cols = [c for c in sales.columns if c.startswith('d_')]\n"
"\n"
"# Фильтрация по штату CA\n"
"sales_ca = sales[sales['state_id'] == 'CA']\n\n"
"# Суммарные продажи по всем товарам\n"
"y = sales_ca[day_cols].sum(axis=0).rename('sales').reset_index()\n"
"y.columns = ['d', 'sales']\n\n"
"# Объединение с календарём\n"
"calendar_small = calendar[['d','date','wday','month','year','snap_CA','event_name_1','event_name_2']]\n"
"df = y.merge(calendar_small, on='d', how='left')\n\n"
"df['date'] = pd.to_datetime(df['date'])\n"
"df = df.sort_values('date').reset_index(drop=True)\n\n"
"df['is_event'] = ((~df['event_name_1'].isna()) | (~df['event_name_2'].isna())).astype(int)\n"
"df.rename(columns={'snap_CA':'snap'}, inplace=True)\n\n"
"df.head()"
))

notebook.cells.append(new_markdown_cell(
"## 4. Feature Engineering\n\n"
"Создаются лаговые и скользящие признаки."
))

notebook.cells.append(new_code_cell(
"df['t'] = np.arange(len(df))\n\n"
"for lag in [1,7,14,28]:\n"
"    df[f'lag_{lag}'] = df['sales'].shift(lag)\n\n"
"for win in [7,28]:\n"
"    df[f'roll_mean_{win}'] = df['sales'].shift(1).rolling(win).mean()\n"
"    df[f'roll_std_{win}'] = df['sales'].shift(1).rolling(win).std()\n\n"
"df = df.dropna().reset_index(drop=True)\n"
"df.head()"
))

notebook.cells.append(new_markdown_cell(
"## 5. Формирование обучающей выборки (One-Model-with-Offset)\n\n"
"Одна модель обучается для всех горизонтов (1–28 дней) с добавлением признака `h`."
))

notebook.cells.append(new_code_cell(
"train_end = len(df) - H - 1\n\n"
"base_cols = ['t','wday','month','year','snap','is_event',\n"
"             'lag_1','lag_7','lag_14','lag_28',\n"
"             'roll_mean_7','roll_mean_28','roll_std_7','roll_std_28']\n\n"
"fut_cols = ['wday','month','year','snap','is_event']\n\n"
"X_list, y_list = [], []\n\n"
"for t in range(MAX_LAG, train_end - H):\n"
"    for h in range(1, H+1):\n"
"        base = df.loc[t, base_cols].values\n"
"        future = df.loc[t+h, fut_cols].values\n"
"        X_list.append(np.concatenate([base, future, [h]]))\n"
"        y_list.append(df.loc[t+h, 'sales'])\n\n"
"X = np.array(X_list)\n"
"y_train = np.array(y_list)\n\n"
"X.shape"
))

notebook.cells.append(new_markdown_cell(
"## 6. Обучение моделей"
))

notebook.cells.append(new_code_cell(
"xgb_model = xgb.XGBRegressor(\n"
"    n_estimators=2000,\n"
"    learning_rate=0.03,\n"
"    max_depth=6,\n"
"    subsample=0.8,\n"
"    colsample_bytree=0.8,\n"
"    objective='reg:squarederror',\n"
"    tree_method='hist',\n"
"    random_state=42\n"
")\n\n"
"xgb_model.fit(X, y_train)\n\n"
"lgb_model = lgb.LGBMRegressor(\n"
"    n_estimators=3000,\n"
"    learning_rate=0.03,\n"
"    num_leaves=64,\n"
"    subsample=0.8,\n"
"    colsample_bytree=0.8,\n"
"    random_state=42\n"
")\n\n"
"lgb_model.fit(X, y_train)"
))

notebook.cells.append(new_markdown_cell(
"## 7. Прогноз на последние 28 дней и расчёт WAPE"
))

notebook.cells.append(new_code_cell(
"X_test = []\n"
"y_true = []\n"
"\n"
"for h in range(1, H+1):\n"
"    base = df.loc[train_end, base_cols].values\n"
"    future = df.loc[train_end+h, fut_cols].values\n"
"    X_test.append(np.concatenate([base, future, [h]]))\n"
"    y_true.append(df.loc[train_end+h, 'sales'])\n\n"
"X_test = np.array(X_test)\n"
"y_true = np.array(y_true)\n\n"
"y_pred_xgb = xgb_model.predict(X_test)\n"
"y_pred_lgb = lgb_model.predict(X_test)\n\n"
"print('WAPE XGBoost:', wape(y_true, y_pred_xgb))\n"
"print('WAPE LightGBM:', wape(y_true, y_pred_lgb))"
))

notebook.cells.append(new_markdown_cell(
"## 8. Визуализация результатов"
))

notebook.cells.append(new_code_cell(
"dates = df.loc[train_end+1:train_end+H, 'date']\n\n"
"plt.figure(figsize=(10,4))\n"
"plt.plot(dates, y_true, label='Actual')\n"
"plt.plot(dates, y_pred_xgb, label='XGBoost')\n"
"plt.plot(dates, y_pred_lgb, label='LightGBM')\n"
"plt.xticks(rotation=45)\n"
"plt.legend()\n"
"plt.title('Forecast vs Actual (28 days)')\n"
"plt.tight_layout()\n"
"plt.show()"
))

notebook.cells.append(new_markdown_cell(
"## 9. Выводы\n\n"
"В данном проекте была разработана система прогнозирования спроса на основе бустинговых моделей.\n\n"
"Проведено сравнение XGBoost и LightGBM по метрике WAPE.\n\n"
"По полученным результатам можно определить модель, обеспечивающую более точный прогноз спроса."
))


# Save notebook
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "Course_Project_M5_XGB_LGBM_CA.ipynb"
with open(file_path, "w", encoding="utf-8") as f:
    nbf.write(notebook, f)

file_path
