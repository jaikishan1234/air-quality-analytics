import pandas as pd, numpy as np

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
print('df_model shape:', df_model.shape)
d_min = str(df_model['Date'].min().date())
d_max = str(df_model['Date'].max().date())
print('Date range:', d_min, '->', d_max)
print('Cities:', df_model['City'].nunique())

split_idx = int(len(df_model) * 0.80)
df_train = df_model.iloc[:split_idx]
df_test  = df_model.iloc[split_idx:]
t_min = str(df_train['Date'].min().date())
t_max = str(df_train['Date'].max().date())
v_min = str(df_test['Date'].min().date())
v_max = str(df_test['Date'].max().date())
print()
print('Total modeling rows:', len(df_model))
print('Train rows:', len(df_train), '  (', t_min, '->', t_max, ')')
print('Test rows :', len(df_test),  '  (', v_min, '->', v_max, ')')
print()
n_cities = df_model['City'].nunique()
print('Cities (one-hot cols):', n_cities)
print('Pollutant features:', len(POLL11))
temporal_feats = ['Year','Month','Day','DayOfWeek','DayOfYear','month_sin','month_cos','doy_sin','doy_cos']
print('Temporal features:', len(temporal_feats))
total_numeric = len(POLL11) + len(temporal_feats)
print('Total features (approx after OHE, drop-first):', total_numeric + n_cities - 1)

# Baseline: median of training AQI
baseline_pred = df_train['AQI'].median()
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_test = df_test['AQI'].values
y_base = np.full_like(y_test, fill_value=baseline_pred, dtype=float)
mae  = mean_absolute_error(y_test, y_base)
rmse = np.sqrt(mean_squared_error(y_test, y_base))
r2   = r2_score(y_test, y_base)
print()
print('Baseline (train median = %.1f) on test set:' % baseline_pred)
print('  MAE : %.2f' % mae)
print('  RMSE: %.2f' % rmse)
print('  R2  : %.4f' % r2)
