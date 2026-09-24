import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

# ── Reproduce cleaned df_model ─────────────────────────────────────────────
df_raw = pd.read_csv('data/city_day.csv')
df = df_raw.copy()
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['DayOfYear'] = df['Date'].dt.dayofyear

POLL12 = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
df = df[~df[POLL12].isnull().all(axis=1)].copy().reset_index(drop=True)
df.drop(columns=['Xylene'], inplace=True)
POLL11 = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene']
df[POLL11] = df.groupby('City')[POLL11].transform(lambda c: c.fillna(c.median()))
df[POLL11] = df[POLL11].fillna(df[POLL11].median())

def assign_season(m):
    if m in [12,1,2]: return 'Winter'
    if m in [3,4,5]: return 'Spring/Pre-Monsoon'
    if m in [6,7,8,9]: return 'Monsoon'
    return 'Post-Monsoon/Autumn'
df['Season'] = df['Month'].map(assign_season)

df_model = df.dropna(subset=['AQI']).copy().sort_values('Date').reset_index(drop=True)
df_model['Day']       = df_model['Date'].dt.day
df_model['DayOfWeek'] = df_model['Date'].dt.dayofweek
df_model['month_sin'] = np.sin(2*np.pi*df_model['Month']/12)
df_model['month_cos'] = np.cos(2*np.pi*df_model['Month']/12)
df_model['doy_sin']   = np.sin(2*np.pi*df_model['DayOfYear']/365)
df_model['doy_cos']   = np.cos(2*np.pi*df_model['DayOfYear']/365)

POLLUTANT_FEATURES = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene']
TEMPORAL_FEATURES  = ['Year','Month','Day','DayOfWeek','DayOfYear','month_sin','month_cos','doy_sin','doy_cos']
NUMERIC_FEATURES   = POLLUTANT_FEATURES + TEMPORAL_FEATURES
CITY_FEATURE       = ['City']
TARGET             = 'AQI'

split_idx = int(len(df_model) * 0.80)
df_train = df_model.iloc[:split_idx].copy()
df_test  = df_model.iloc[split_idx:].copy()

X_train_raw = df_train[NUMERIC_FEATURES + CITY_FEATURE]
X_test_raw  = df_test[NUMERIC_FEATURES  + CITY_FEATURE]
y_train = df_train[TARGET].values
y_test  = df_test[TARGET].values

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('scaler', StandardScaler())]), NUMERIC_FEATURES),
    ('cat', Pipeline([('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))]), CITY_FEATURE)
])
X_train_proc = preprocessor.fit_transform(X_train_raw)
X_test_proc  = preprocessor.transform(X_test_raw)

# ── Train models ──────────────────────────────────────────────────────────────
models = {
    'Baseline (Dummy)':        DummyRegressor(strategy='median'),
    'Linear Regression':       LinearRegression(),
    'Ridge':                   Ridge(alpha=10.0, random_state=42),
    'Random Forest':           RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1),
    'Gradient Boosting':       GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    'HistGradient Boosting':   HistGradientBoostingRegressor(max_iter=200, max_depth=8, learning_rate=0.1, random_state=42),
}

results = []
preds   = {}
for name, model in models.items():
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)
    preds[name] = y_pred
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    maxae = np.max(np.abs(y_test - y_pred))
    results.append({'Model': name, 'MAE': round(mae,2), 'RMSE': round(rmse,2),
                    'R2': round(r2,4), 'MedAE': round(medae,2), 'MaxAE': round(maxae,2)})
    print(f'{name:30s} MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}')

df_res = pd.DataFrame(results).sort_values('RMSE')
print()
print(df_res.to_string(index=False))

# Best by RMSE
best_name = df_res[df_res['Model'] != 'Baseline (Dummy)'].iloc[0]['Model']
print()
print('Best model by RMSE:', best_name)

# Baseline stats
base_row = df_res[df_res['Model']=='Baseline (Dummy)'].iloc[0]
print()
print('Improvement vs baseline for best model:')
best_row = df_res[df_res['Model']==best_name].iloc[0]
for m in ['MAE','RMSE']:
    imp = (base_row[m] - best_row[m]) / base_row[m] * 100
    print(f'  {m}: {base_row[m]:.2f} -> {best_row[m]:.2f}  ({imp:.1f}% improvement)')
print(f'  R2: {base_row["R2"]:.4f} -> {best_row["R2"]:.4f}')
