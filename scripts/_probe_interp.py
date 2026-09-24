import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.inspection import permutation_importance

# ── Rebuild cleaned df_model ──────────────────────────────────────────────────
df_raw = pd.read_csv('data/city_day.csv')
df = df_raw.copy()
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year;  df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter;  df['DayOfYear'] = df['Date'].dt.dayofyear

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
df_model['Day'] = df_model['Date'].dt.day
df_model['DayOfWeek'] = df_model['Date'].dt.dayofweek
df_model['month_sin'] = np.sin(2*np.pi*df_model['Month']/12)
df_model['month_cos'] = np.cos(2*np.pi*df_model['Month']/12)
df_model['doy_sin'] = np.sin(2*np.pi*df_model['DayOfYear']/365)
df_model['doy_cos'] = np.cos(2*np.pi*df_model['DayOfYear']/365)

POLL_FEATS = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene']
TEMP_FEATS = ['Year','Month','Day','DayOfWeek','DayOfYear','month_sin','month_cos','doy_sin','doy_cos']
NUMERIC_FEATURES = POLL_FEATS + TEMP_FEATS
CITY_FEATURE = ['City'];  TARGET = 'AQI'

split_idx = int(len(df_model)*0.80)
df_train = df_model.iloc[:split_idx].copy();  df_test = df_model.iloc[split_idx:].copy()
X_train_raw = df_train[NUMERIC_FEATURES+CITY_FEATURE];  X_test_raw = df_test[NUMERIC_FEATURES+CITY_FEATURE]
y_train = df_train[TARGET].values;  y_test = df_test[TARGET].values

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('scaler', StandardScaler())]), NUMERIC_FEATURES),
    ('cat', Pipeline([('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))]), CITY_FEATURE)
])
X_train_proc = preprocessor.fit_transform(X_train_raw)
X_test_proc  = preprocessor.transform(X_test_raw)

ohe_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(CITY_FEATURE)
all_feat_names = NUMERIC_FEATURES + list(ohe_cols)

best_model = HistGradientBoostingRegressor(max_iter=200, max_depth=8, learning_rate=0.1, random_state=42)
best_model.fit(X_train_proc, y_train)
y_pred = best_model.predict(X_test_proc)
residuals = y_test - y_pred

print('Best model metrics:')
print('  MAE :', round(mean_absolute_error(y_test, y_pred), 2))
print('  RMSE:', round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print('  R2  :', round(r2_score(y_test, y_pred), 4))
print('  MedAE:', round(median_absolute_error(y_test, y_pred), 2))
print('  MaxAE:', round(np.max(np.abs(residuals)), 2))

# Permutation importance (n_repeats=10, reduced for speed)
print('\nRunning permutation importance...')
perm = permutation_importance(best_model, X_test_proc, y_test,
                              n_repeats=10, random_state=42, scoring='r2', n_jobs=-1)
perm_df = pd.DataFrame({
    'Feature': all_feat_names,
    'Importance_Mean': perm.importances_mean.round(4),
    'Importance_Std':  perm.importances_std.round(4),
}).sort_values('Importance_Mean', ascending=False).reset_index(drop=True)
print('\nTop 20 features by permutation importance:')
print(perm_df.head(20).to_string())

# Group by type
poll_mask = perm_df['Feature'].isin(POLL_FEATS)
temp_mask = perm_df['Feature'].isin(TEMP_FEATS)
city_mask = perm_df['Feature'].str.startswith('City_')
print('\nGroup totals:')
print('  Pollutant total:', round(perm_df[poll_mask]['Importance_Mean'].sum(), 4))
print('  Temporal total :', round(perm_df[temp_mask]['Importance_Mean'].sum(), 4))
print('  City total     :', round(perm_df[city_mask]['Importance_Mean'].sum(), 4))

# AQI range errors
abs_err = np.abs(residuals)
df_err = df_test[['Date','City','AQI']].copy().reset_index(drop=True)
df_err['Predicted'] = y_pred.round(1)
df_err['Residual']  = residuals.round(1)
df_err['AbsError']  = abs_err.round(1)

def aqi_cat(a):
    if a <= 50: return 'Good (0-50)'
    if a <= 100: return 'Satisfactory (51-100)'
    if a <= 200: return 'Moderate (101-200)'
    if a <= 300: return 'Poor (201-300)'
    if a <= 400: return 'Very Poor (301-400)'
    return 'Severe (401+)'
df_err['AQI_Range'] = df_err['AQI'].apply(aqi_cat)
range_stats = df_err.groupby('AQI_Range').apply(
    lambda g: pd.Series({'N': len(g), 'MAE': g['AbsError'].mean().round(2),
                         'RMSE': np.sqrt((g['Residual']**2).mean()).round(2)}), include_groups=False
).reset_index()
print('\nAQI Range Error:')
print(range_stats)

# City errors
city_stats = df_err.groupby('City').apply(
    lambda g: pd.Series({'N': len(g), 'MAE': g['AbsError'].mean().round(2),
                         'RMSE': np.sqrt((g['Residual']**2).mean()).round(2),
                         'Mean_Actual': g['AQI'].mean().round(1),
                         'Mean_Pred': g['Predicted'].mean().round(1)}), include_groups=False
).sort_values('MAE', ascending=False)
print('\nCity Error stats (top 10):')
print(city_stats.head(10))

# Temporal errors (monthly)
df_err['Month'] = df_test['Date'].dt.month.values
monthly_err = df_err.groupby('Month').apply(
    lambda g: pd.Series({'N': len(g), 'MAE': g['AbsError'].mean().round(2),
                         'RMSE': np.sqrt((g['Residual']**2).mean()).round(2)}), include_groups=False
)
print('\nMonthly errors on test set:')
print(monthly_err)
