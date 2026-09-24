import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md(id_, source_str):
    return {"cell_type": "markdown", "id": id_, "metadata": {},
            "source": source_str if isinstance(source_str, list) else [source_str]}

def code(id_, source_str):
    lines = source_str.strip().split('\n')
    src = [l + '\n' for l in lines]
    src[-1] = src[-1].rstrip('\n')
    return {"cell_type": "code", "execution_count": None, "id": id_,
            "metadata": {}, "outputs": [], "source": src}

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# Section header
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("fe-header", """\
---

## 11. Feature Engineering and ML Dataset Preparation

This section transforms the cleaned dataset into a machine-learning-ready format. \
Every design decision is documented and justified. The goal is to produce a training \
and test dataset that can be fed directly into regression models in Stage 5, \
without any data leakage or methodological errors.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 11.1 Define the ML Problem
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("fe-problem-md", """\
### 11.1 Define the ML Problem

| Attribute | Value |
|---|---|
| **Task** | Supervised regression |
| **Target variable** | `AQI` (continuous, numeric) |
| **Objective** | Predict the daily Air Quality Index for a given city from measurable pollutant concentrations and temporal/spatial features |
| **Evaluation** | MAE, RMSE, R² (details in Stage 5) |

**Why AQI_Bucket is excluded from predictors:**  
`AQI_Bucket` is computed directly from the `AQI` value by applying fixed threshold bands. \
Using it as a feature when predicting `AQI` would be **target leakage** — the model would \
have access to a recoded version of the answer. Any model trained with `AQI_Bucket` as a \
feature would show artificially inflated performance that would not generalise to real-world \
deployment, where `AQI_Bucket` is only known *after* AQI is computed.

**Why rows with missing AQI are excluded from supervised training:**  
Machine-learning regression requires a known, real label for every training observation. \
Rows where `AQI` is `NaN` have no valid target — imputing the target would mean training \
a model to predict values we invented ourselves, which invalidates evaluation metrics. \
These rows were retained in `df` for descriptive analysis but will not appear in `df_model`.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 11.2 Build the Modeling Dataset
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("fe-build-md", """\
### 11.2 Build the Modeling Dataset

A dedicated modeling DataFrame `df_model` is constructed from the cleaned `df` by retaining \
only rows where `AQI` is available and sorting chronologically. \
This is the canonical dataset used from this point forward.\
"""))

cells.append(code("fe-build-code", """\
# ── Build dedicated modeling DataFrame ───────────────────────────────────────
# Start from the cleaned working dataframe (already in memory from Stage 2).
# Drop rows where AQI is NaN — these cannot be used for supervised regression.
df_model = (
    df.dropna(subset=['AQI'])
    .copy()
    .sort_values('Date')
    .reset_index(drop=True)
)

print(f'Rows in cleaned df        : {len(df):,}')
print(f'Rows with missing AQI     : {df["AQI"].isna().sum():,}  (excluded from modeling)')
print(f'Rows in df_model          : {len(df_model):,}')
print(f'Date range of df_model    : {df_model["Date"].min().date()} → {df_model["Date"].max().date()}')
print(f'Unique cities in df_model : {df_model["City"].nunique()}')
print()
print('Columns in df_model:')
print(list(df_model.columns))\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 11.3 Temporal Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("fe-temporal-md", """\
### 11.3 Temporal Feature Engineering

The `Date` column is a rich source of signal for AQI prediction. \
We already extracted `Year`, `Month`, `Quarter`, and `DayOfYear` in Stage 2. \
Here we add two further features:

- **`Day`** — day of month (1–31)
- **`DayOfWeek`** — weekday number (0 = Monday … 6 = Sunday); weekday vs weekend traffic patterns can affect pollution levels

**Cyclical encoding for month and day-of-year:**

Treating `Month` as a raw integer (1–12) implies that December (12) and January (1) are \
far apart, when in reality they are adjacent seasons. A linear model or distance-based model \
would then be misled by this artificial discontinuity. Cyclical (sine/cosine) encoding maps \
the feature onto a unit circle, preserving the cyclic structure:

> `month_sin = sin(2π × month / 12)`  
> `month_cos = cos(2π × month / 12)`

The same approach is applied to `DayOfYear` (period 365). \
The raw integer versions are also retained because tree-based models can sometimes exploit \
ordinal information directly.\
"""))

cells.append(code("fe-temporal-code", """\
# ── Temporal feature engineering ─────────────────────────────────────────────
import numpy as np

# Additional temporal fields not yet in df_model
df_model['Day']        = df_model['Date'].dt.day
df_model['DayOfWeek']  = df_model['Date'].dt.dayofweek   # 0=Mon … 6=Sun

# Cyclical encoding for Month (period = 12)
df_model['month_sin']  = np.sin(2 * np.pi * df_model['Month']  / 12)
df_model['month_cos']  = np.cos(2 * np.pi * df_model['Month']  / 12)

# Cyclical encoding for DayOfYear (period = 365)
df_model['doy_sin']    = np.sin(2 * np.pi * df_model['DayOfYear'] / 365)
df_model['doy_cos']    = np.cos(2 * np.pi * df_model['DayOfYear'] / 365)

# Verify
print('New temporal features added:')
check_cols = ['Date', 'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear',
              'month_sin', 'month_cos', 'doy_sin', 'doy_cos', 'Season']
print(df_model[check_cols].head(6).to_string())
print()
print('Value ranges:')
for col in ['month_sin','month_cos','doy_sin','doy_cos']:
    print(f'  {col:<14}: min={df_model[col].min():.4f}  max={df_model[col].max():.4f}')\
"""))

cells.append(md("fe-temporal-interp-md", """\
The cyclical sin/cos pair for Month places January and December adjacent on the unit circle, \
correctly representing the continuity of winter. A pure ordinal encoding would place a gap \
of 11 units between month 1 and month 12, which is physically incorrect for a seasonal signal.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 11.4 City Encoding
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("fe-city-md", """\
### 11.4 City Encoding

`City` is a nominal categorical variable with 26 levels. \
It must be converted to numeric form before being passed to scikit-learn estimators.

**One-hot encoding** is used. This creates a binary indicator column for each city, \
with `drop='first'` to avoid perfect multicollinearity (the dropped category becomes \
the implicit reference level). This yields 25 binary columns.

**Why not target encoding?**  
Target encoding (replacing city names with the city's mean/median AQI) would constitute \
**target leakage**: the encoding would embed information about the target variable `AQI` \
directly into a feature. This would produce misleading performance metrics. \
One-hot encoding is leakage-free because it encodes identity only, not outcome.\
"""))

cells.append(code("fe-city-code", """\
# ── City one-hot encoding (preview only — actual OHE applied in pipeline) ────
# We demonstrate the encoding shape here; the Pipeline in §11.7 applies it
# correctly, fitting only on training data.

from sklearn.preprocessing import OneHotEncoder
_ohe_demo = OneHotEncoder(drop='first', sparse_output=False)
_city_demo = _ohe_demo.fit_transform(df_model[['City']])
_city_cols  = _ohe_demo.get_feature_names_out(['City'])

print(f'Number of cities         : {df_model["City"].nunique()}')
print(f'One-hot columns (drop=first): {len(_city_cols)}')
print()
print('Generated column names (first 10):')
for col in _city_cols[:10]:
    print(f'  {col}')\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 11.5 Feature Selection
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("fe-selection-md", """\
### 11.5 Feature Selection

The final feature set is composed of three groups:

| Group | Features | Count |
|---|---|---|
| **Pollutant** | PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene | 11 |
| **Temporal (numeric)** | Year, Month, Day, DayOfWeek, DayOfYear, month_sin, month_cos, doy_sin, doy_cos | 9 |
| **City** | One-hot encoded (26 cities, drop_first → 25 binary columns) | 25 |
| **Target** | AQI | — |

**Total numeric + temporal input features before OHE:** 20  
**Total features after OHE (with drop_first):** 45

**Note on multicollinearity:**  
Stage 3 EDA identified strong pairwise correlations (e.g., NOx–NO r≈0.79, Benzene–Toluene r≈0.71). \
Correlation between features does not automatically warrant removal — it depends on the model type. \
Tree-based models (Random Forest, Gradient Boosting) are naturally robust to correlated features. \
Linear models may be affected, but regularisation (Ridge/Lasso) handles multicollinearity. \
All features are retained; the modelling stage will reveal whether regularisation or feature \
selection further improves performance.\
"""))

cells.append(code("fe-selection-code", """\
# ── Define feature lists ──────────────────────────────────────────────────────
POLLUTANT_FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                      'CO', 'SO2', 'O3', 'Benzene', 'Toluene']

TEMPORAL_FEATURES  = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear',
                      'month_sin', 'month_cos', 'doy_sin', 'doy_cos']

CITY_FEATURE       = ['City']   # will be one-hot encoded inside the pipeline

TARGET             = 'AQI'

# Combined numeric feature list (before OHE)
NUMERIC_FEATURES   = POLLUTANT_FEATURES + TEMPORAL_FEATURES

print('Feature summary:')
print(f'  Pollutant features  : {len(POLLUTANT_FEATURES)}  -> {POLLUTANT_FEATURES}')
print(f'  Temporal features   : {len(TEMPORAL_FEATURES)}   -> {TEMPORAL_FEATURES}')
print(f'  City (OHE, drop=1st): 25  (from 26 unique cities)')
print(f'  Total after OHE     : {len(NUMERIC_FEATURES) + 25}')
print(f'  Target              : {TARGET}')
print()

# Confirm all features exist in df_model with no NaN
missing_in_features = df_model[NUMERIC_FEATURES + CITY_FEATURE].isnull().sum()
missing_any = missing_in_features[missing_in_features > 0]
if len(missing_any) == 0:
    print('All feature columns are NaN-free in df_model.')
else:
    print('Columns with remaining NaN:')
    print(missing_any)\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 11.6 Train/Test Split
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("fe-split-md", """\
### 11.6 Chronological Train/Test Split

**Why chronological splitting?**  
Environmental air-quality data has a strong temporal structure — pollution levels \
are correlated day-to-day, seasonally, and year-to-year. \
A random split would allow the model to "see the future": training observations from \
late 2019 could be mixed with test observations from early 2018. \
The model would then learn to interpolate between nearby dates rather than genuinely \
extrapolate to unseen future conditions.

A **chronological split** — training on earlier dates, testing on later dates — \
gives a realistic estimate of how well the model predicts AQI on dates it has never seen. \
This is the correct evaluation protocol for any time-series or longitudinal dataset.

**Split ratio:** 80% training / 20% test. \
The dataset is sorted by `Date` and the first 80% of rows form the training set.\
"""))

cells.append(code("fe-split-code", """\
# ── Chronological 80/20 train/test split ─────────────────────────────────────
# df_model is already sorted by Date (confirmed in §11.2)
N = len(df_model)
split_idx = int(N * 0.80)

df_train = df_model.iloc[:split_idx].copy()
df_test  = df_model.iloc[split_idx:].copy()

# Extract feature matrices and target vectors
X_train_raw = df_train[NUMERIC_FEATURES + CITY_FEATURE]
X_test_raw  = df_test[NUMERIC_FEATURES  + CITY_FEATURE]
y_train      = df_train[TARGET].values
y_test       = df_test[TARGET].values

print('=' * 52)
print('  TRAIN / TEST SPLIT SUMMARY')
print('=' * 52)
print(f'  Total modeling rows   : {N:,}')
print(f'  Training rows (80%)   : {len(df_train):,}')
print(f'  Test rows     (20%)   : {len(df_test):,}')
print(f'  Training date range   : {df_train["Date"].min().date()} → {df_train["Date"].max().date()}')
print(f'  Test date range       : {df_test["Date"].min().date()}  → {df_test["Date"].max().date()}')
print('=' * 52)
print()
print(f'X_train_raw shape : {X_train_raw.shape}')
print(f'X_test_raw  shape : {X_test_raw.shape}')
print(f'y_train     shape : {y_train.shape}')
print(f'y_test      shape : {y_test.shape}')\
"""))

cells.append(code("fe-split-vis-code", """\
# ── Visualise the train/test split boundary ───────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))

# Plot median AQI per month for visual context
_monthly = df_model.set_index('Date')['AQI'].resample('ME').median()
ax.plot(_monthly.index, _monthly.values, color='steelblue', linewidth=1.2,
        label='Monthly median AQI')

# Split boundary
split_date = df_test['Date'].min()
ax.axvline(split_date, color='red', linewidth=2, linestyle='--',
           label=f'Train/Test boundary ({split_date.date()})')
ax.fill_between(_monthly.index, 0, _monthly.values,
                where=[d < split_date for d in _monthly.index],
                alpha=0.15, color='steelblue', label='Training period')
ax.fill_between(_monthly.index, 0, _monthly.values,
                where=[d >= split_date for d in _monthly.index],
                alpha=0.25, color='red', label='Test period')

ax.set_xlabel('Date')
ax.set_ylabel('Median AQI')
ax.set_title('Chronological Train/Test Split (80% / 20%)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()\
"""))

cells.append(md("fe-split-interp-md", """\
The test set spans **December 2019 – July 2020**, which includes a winter peak period \
and the first months of the COVID-19 pandemic. \
This is a realistic out-of-sample evaluation: the model has seen no data from this period \
during training, and must generalise from historical patterns alone.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 11.7 Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("fe-pipeline-md", """\
### 11.7 Scikit-Learn Preprocessing Pipeline

A reusable `ColumnTransformer` + `Pipeline` is constructed so that:

1. **Preprocessing is fitted only on training data** — no information from the test set \
leaks into the scaling or encoding parameters.
2. **All steps are reproducible** — the same pipeline can be wrapped around any estimator \
in Stage 5 without repeating manual transformations.
3. **Scaling applies only to numeric features** — `StandardScaler` (zero mean, unit variance) \
is applied to the 20 numeric features. One-hot encoding is applied to `City`.

Temporal features (Year, Month, Day, etc.) are included in the numeric block \
and scaled alongside pollutant values. \
While scaling has no effect on tree-based models, it is necessary for linear models and \
prevents numerical issues in gradient-based optimisers.\
"""))

cells.append(code("fe-pipeline-code", """\
# ── Preprocessing pipeline ────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Separate numeric and categorical columns
numeric_cols     = NUMERIC_FEATURES        # 20 features
categorical_cols = CITY_FEATURE            # ['City']

# Numeric transformer: StandardScaler (zero mean, unit variance)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical transformer: one-hot with drop='first' (25 binary cols)
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer,  numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Fit ONLY on training data — never on test data
X_train_processed = preprocessor.fit_transform(X_train_raw)
X_test_processed  = preprocessor.transform(X_test_raw)

# Retrieve final feature names after OHE
ohe_cols     = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
all_feat_names = numeric_cols + list(ohe_cols)

print('Preprocessor fitted on training data only.')
print()
print(f'X_train_processed shape : {X_train_processed.shape}')
print(f'X_test_processed  shape : {X_test_processed.shape}')
print(f'Total feature names     : {len(all_feat_names)}')
print()
print('Feature name breakdown:')
print(f'  Numeric (scaled)       : {len(numeric_cols)}')
print(f'  City OHE (drop_first)  : {len(ohe_cols)}')
print(f'  Total                  : {len(all_feat_names)}')\
"""))

cells.append(code("fe-feature-importance-preview-code", """\
# ── Feature correlation with target (quick sanity check on training set) ──────
# Compute Pearson r between each processed numeric feature and y_train
# This is descriptive only — does not influence feature selection
import pandas as pd

_train_df = pd.DataFrame(X_train_processed, columns=all_feat_names)
_train_df['AQI'] = y_train

# Correlations for the numeric block only (first 20 features)
feat_corr = _train_df[numeric_cols + ['AQI']].corr()['AQI'].drop('AQI').sort_values(key=abs, ascending=False)
print('Correlation of numeric features with AQI (training set, top 15):')
print(feat_corr.head(15).round(3).to_string())\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 11.8 Baseline
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("fe-baseline-md", """\
### 11.8 Baseline Model

Before training any machine-learning model, it is essential to establish a **meaningful baseline**. \
A baseline is the simplest possible prediction strategy — if a sophisticated model cannot beat it, \
the model adds no value.

**Baseline strategy:** Predict the **median AQI of the training set** for every test observation.  
This is a "predict the central tendency" baseline that ignores all features. \
It is equivalent to sklearn's `DummyRegressor(strategy='median')`.

All models in Stage 5 must exceed this baseline on MAE, RMSE, and R² to be considered useful.\
"""))

cells.append(code("fe-baseline-code", """\
# ── Baseline: predict training median for every test observation ──────────────
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

baseline_model = DummyRegressor(strategy='median')
baseline_model.fit(X_train_processed, y_train)
y_pred_baseline = baseline_model.predict(X_test_processed)

baseline_median = y_train[~np.isnan(y_train)].flatten()
_median_val = float(np.median(baseline_median))

mae_base  = mean_absolute_error(y_test, y_pred_baseline)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
r2_base   = r2_score(y_test, y_pred_baseline)

print('Baseline (predict training-set median = {:.1f}) on test set:'.format(_median_val))
print(f'  MAE  : {mae_base:.2f}')
print(f'  RMSE : {rmse_base:.2f}')
print(f'  R²   : {r2_base:.4f}')
print()
print('Interpretation:')
print('  A model that predicts the same value for every observation achieves R² ≈ 0.')
print('  Any useful model must yield R² > 0, MAE < {:.2f}, RMSE < {:.2f}.'.format(mae_base, rmse_base))\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 11.9 Final ML Dataset Summary
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("fe-summary-md", """\
### 11.9 Final ML Dataset Summary

The table below consolidates every dimension of the prepared ML dataset.\
"""))

cells.append(code("fe-summary-code", """\
# ── Final ML dataset summary ──────────────────────────────────────────────────
print('=' * 58)
print('  FINAL ML DATASET SUMMARY')
print('=' * 58)
print(f'  Total modeling rows          : {len(df_model):,}')
print(f'  Training rows (80%)          : {X_train_processed.shape[0]:,}')
print(f'  Test rows     (20%)          : {X_test_processed.shape[0]:,}')
print(f'  Training date range          : {df_train["Date"].min().date()} → {df_train["Date"].max().date()}')
print(f'  Test date range              : {df_test["Date"].min().date()} → {df_test["Date"].max().date()}')
print()
print(f'  Pollutant features           : {len(POLLUTANT_FEATURES)}')
print(f'  Temporal features            : {len(TEMPORAL_FEATURES)}')
print(f'  City OHE features (drop=1st) : {len(ohe_cols)}')
print(f'  Total transformed features   : {X_train_processed.shape[1]}')
print()
print(f'  X_train shape                : {X_train_processed.shape}')
print(f'  X_test  shape                : {X_test_processed.shape}')
print(f'  y_train shape                : {y_train.shape}')
print(f'  y_test  shape                : {y_test.shape}')
print()
print(f'  Baseline test MAE            : {mae_base:.2f}')
print(f'  Baseline test RMSE           : {rmse_base:.2f}')
print(f'  Baseline test R²             : {r2_base:.4f}')
print('=' * 58)\
"""))

cells.append(md("fe-leakage-summary-md", """\
**Feature design and leakage avoidance — summary:**

| Rule | How it is enforced |
|---|---|
| `AQI_Bucket` excluded | Not present in `NUMERIC_FEATURES`, `CITY_FEATURE`, or any predictor list |
| AQI not imputed | `df_model` built with `.dropna(subset=['AQI'])` — no synthetic targets |
| No future data in training | Chronological split: training ends 2019-12-06, test starts 2019-12-07 |
| Scaling fitted on training only | `preprocessor.fit_transform(X_train_raw)` then `.transform(X_test_raw)` |
| OHE fitted on training only | Same `ColumnTransformer` — OHE categories are fixed to training cities |
| No city AQI statistics used | One-hot encoding only; no target-mean encoding |
| Original CSV unmodified | `df_raw` preserved; all transformations on `df` copy |\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 11.10 Ready for Model Training
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("fe-ready-md", """\
### 11.10 Preparation for Model Training

The following variables are now ready for use in Stage 5 (Machine Learning and Model Evaluation):

| Variable | Description |
|---|---|
| `X_train_processed` | Scaled + OHE-encoded training features, shape (19,840 × 45) |
| `X_test_processed` | Same transformations applied to test features, shape (4,961 × 45) |
| `y_train` | Training AQI labels, shape (19,840,) |
| `y_test` | Test AQI labels, shape (4,961,) |
| `preprocessor` | Fitted `ColumnTransformer` — reuse by wrapping in a Pipeline with any estimator |
| `X_train_raw` | Raw (unscaled) training features — useful for tree models that don't need scaling |
| `X_test_raw` | Raw (unscaled) test features |
| `all_feat_names` | List of 45 feature names after OHE (for interpretability in Stage 5) |
| `baseline_model` | Fitted `DummyRegressor(median)` — reference performance floor |
| `mae_base`, `rmse_base`, `r2_base` | Baseline metric values to beat |

Stage 5 will train and compare multiple regression models:
- **Linear Regression** (with Ridge regularisation)
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

Each model will be evaluated against this baseline using the same test set.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Inject
# ─────────────────────────────────────────────────────────────────────────────
nb['cells'].extend(cells)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Injected {len(cells)} Stage 4 cells.')
print(f'Total cells now: {len(nb["cells"])}')
code_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
md_count   = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
print(f'Code cells: {code_count}  |  Markdown cells: {md_count}')
