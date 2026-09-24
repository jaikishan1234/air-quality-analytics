import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md(id_, s):
    return {"cell_type": "markdown", "id": id_, "metadata": {},
            "source": [l + '\n' for l in s.strip().split('\n')][:-1] +
                      [s.strip().split('\n')[-1]]}

def code(id_, s):
    lines = s.strip().split('\n')
    src = [l + '\n' for l in lines]
    src[-1] = src[-1].rstrip('\n')
    return {"cell_type": "code", "execution_count": None, "id": id_,
            "metadata": {}, "outputs": [], "source": src}

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# Section header
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("ml-header", """\
---

## 12. AQI Prediction Models

This section trains, evaluates, and interprets multiple regression models for AQI prediction. \
All models are evaluated on the same chronological held-out test set established in Stage 4 \
(December 2019 – July 2020), using `X_train_processed` / `X_test_processed` and `y_train` / `y_test` \
that are already defined in memory from the previous stage. \
No new splits, no reordering, no target leakage.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 12.1 Model Selection
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("ml-selection-md", """\
### 12.1 Model Selection

A single algorithm is rarely optimal for an unknown problem. \
Comparing a diverse set of models reveals the nature of the data — whether the \
AQI–pollutant relationship is predominantly linear or requires non-linear approximation — \
and provides a defensible basis for algorithm choice.

Six models are trained, ranging from the trivial baseline to expressive ensemble methods:

| # | Model | Type | Notes |
|---|---|---|---|
| 1 | **DummyRegressor (Baseline)** | Trivial | Predicts training-set median; no feature use |
| 2 | **Linear Regression** | Linear | OLS; assumes linear AQI–feature relationship |
| 3 | **Ridge Regression** | Linear + L2 regularisation | Reduces multicollinearity sensitivity |
| 4 | **Random Forest** | Non-linear ensemble | Bagged decision trees; robust to outliers |
| 5 | **Gradient Boosting** | Non-linear ensemble (boosting) | Sequential residual correction |
| 6 | **HistGradient Boosting** | Non-linear (histogram-based boosting) | Faster large-dataset variant of GBM |

All models are trained with `random_state=42` for reproducibility. \
No hyperparameter search is performed at this stage — default-to-reasonable settings are used. \
The goal is a fair, head-to-head comparison before any tuning.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 12.2 Train Models
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("ml-train-md", """\
### 12.2 Train the Models\
"""))

cells.append(code("ml-train-code", """\
# ── Import model classes ──────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingRegressor,
                               HistGradientBoostingRegressor)
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, median_absolute_error)
import time

# ── Model registry ────────────────────────────────────────────────────────────
model_registry = {
    'Baseline (Dummy)':       DummyRegressor(strategy='median'),
    'Linear Regression':      LinearRegression(),
    'Ridge':                  Ridge(alpha=10.0, random_state=42),
    'Random Forest':          RandomForestRegressor(
                                  n_estimators=200, max_depth=20,
                                  min_samples_leaf=2, random_state=42, n_jobs=-1),
    'Gradient Boosting':      GradientBoostingRegressor(
                                  n_estimators=200, max_depth=5,
                                  learning_rate=0.1, random_state=42),
    'HistGradient Boosting':  HistGradientBoostingRegressor(
                                  max_iter=200, max_depth=8,
                                  learning_rate=0.1, random_state=42),
}

# ── Train all models on the training split ────────────────────────────────────
trained_models = {}
train_times    = {}

print(f'{"Model":<26}  {"Train time (s)":>14}')
print('-' * 44)
for name, model in model_registry.items():
    t0 = time.time()
    model.fit(X_train_processed, y_train)
    elapsed = time.time() - t0
    trained_models[name] = model
    train_times[name]    = round(elapsed, 2)
    print(f'{name:<26}  {elapsed:>13.2f}s')

print()
print(f'Models trained on {X_train_processed.shape[0]:,} observations with {X_train_processed.shape[1]} features.')\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 12.3 Evaluate All Models
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("ml-eval-md", """\
### 12.3 Evaluate All Models

Every trained model is applied to the held-out test set. \
Three standard regression metrics are computed:

| Metric | Meaning | Ideal |
|---|---|---|
| **MAE** | Mean Absolute Error — average prediction error in AQI units | Lower |
| **RMSE** | Root Mean Squared Error — penalises large errors more than MAE | Lower |
| **R²** | Coefficient of determination — fraction of AQI variance explained | Higher (max 1.0) |\
"""))

cells.append(code("ml-eval-code", """\
# ── Evaluate all models on the test set ──────────────────────────────────────
results_list = []
test_predictions = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test_processed)
    test_predictions[name] = y_pred

    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    maxae = float(np.max(np.abs(y_test - y_pred)))

    results_list.append({
        'Model':  name,
        'MAE':    round(mae,  2),
        'RMSE':   round(rmse, 2),
        'R2':     round(r2,   4),
        'MedAE':  round(medae,2),
        'MaxAE':  round(maxae, 2),
    })

# Build results DataFrame, sorted by RMSE ascending
model_results = pd.DataFrame(results_list).sort_values('RMSE').reset_index(drop=True)

print('Model Evaluation Results (sorted by RMSE ascending):')
print('=' * 73)
print(model_results[['Model','MAE','RMSE','R2','MedAE','MaxAE']].to_string(index=False))
print('=' * 73)
print()
print('Note: table sorted by RMSE for readability — ranking by a single metric')
print('may differ from ranking by MAE or R².')\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 12.4 Model Comparison Visualisation
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("ml-viz-md", """\
### 12.4 Model Comparison Visualisation\
"""))

cells.append(code("ml-viz-code", """\
# ── Bar-chart comparison across MAE, RMSE, R² ────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

# Use display order: Baseline first, then ascending RMSE for non-baseline
non_base   = model_results[model_results['Model'] != 'Baseline (Dummy)']
base_row   = model_results[model_results['Model'] == 'Baseline (Dummy)']
plot_order = pd.concat([base_row, non_base])['Model'].tolist()

palette = sns.color_palette('muted', len(plot_order))
palette[0] = (0.75, 0.75, 0.75)  # grey for baseline

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

metrics_cfg = [
    ('MAE',  'Mean Absolute Error (AQI units)', 'lower is better'),
    ('RMSE', 'Root Mean Squared Error (AQI units)', 'lower is better'),
    ('R2',   'R² — Variance Explained', 'higher is better'),
]

for ax, (metric, ylabel, note) in zip(axes, metrics_cfg):
    vals   = [plot_order.index(n) for n in plot_order]  # position mapping
    data   = plot_order
    metric_vals = [model_results.loc[model_results['Model']==m, metric].values[0]
                   for m in plot_order]
    colors = palette
    bars = ax.barh(data, metric_vals, color=colors, edgecolor='white')
    for bar, v in zip(bars, metric_vals):
        ax.text(v + (max(metric_vals)*0.01), bar.get_y() + bar.get_height()/2,
                f'{v:.2f}', va='center', fontsize=8.5)
    ax.set_xlabel(ylabel, fontsize=9)
    ax.set_title(f'{metric}  ({note})', fontsize=10)
    ax.set_xlim(0, max(metric_vals) * 1.18)
    ax.invert_yaxis()

plt.suptitle('Regression Model Comparison on AQI Test Set (Dec 2019 – Jul 2020)',
             y=1.02, fontsize=12)
plt.tight_layout()
plt.show()\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 12.5 Compare Against Baseline
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("ml-baseline-md", """\
### 12.5 Improvement Over Baseline\
"""))

cells.append(code("ml-baseline-code", """\
# ── Improvement relative to DummyRegressor baseline ─────────────────────────
baseline = model_results[model_results['Model'] == 'Baseline (Dummy)'].iloc[0]

improvement_rows = []
for _, row in model_results[model_results['Model'] != 'Baseline (Dummy)'].iterrows():
    mae_imp  = baseline['MAE']  - row['MAE']
    rmse_imp = baseline['RMSE'] - row['RMSE']
    r2_imp   = row['R2']        - baseline['R2']
    mae_pct  = mae_imp  / baseline['MAE']  * 100
    rmse_pct = rmse_imp / baseline['RMSE'] * 100
    improvement_rows.append({
        'Model':              row['Model'],
        'MAE (test)':         row['MAE'],
        'MAE △ vs baseline':  f'-{mae_imp:.2f}  (-{mae_pct:.1f}%)',
        'RMSE (test)':        row['RMSE'],
        'RMSE △ vs baseline': f'-{rmse_imp:.2f}  (-{rmse_pct:.1f}%)',
        'R² (test)':          row['R2'],
        'R² △ vs baseline':   f'+{r2_imp:.4f}',
    })

improvement_df = pd.DataFrame(improvement_rows)
print(f'Baseline: MAE={baseline["MAE"]:.2f}  RMSE={baseline["RMSE"]:.2f}  R²={baseline["R2"]:.4f}')
print()
print('Improvement of each model over DummyRegressor (Baseline):')
print(improvement_df.to_string(index=False))\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 12.6 Model Interpretation (Markdown — filled in after probe results)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("ml-interp-md", """\
### 12.6 Model Interpretation

The following observations are grounded in the metric values computed above.

**All ML models substantially outperform the baseline.**  
Even the weakest learner, Linear Regression (MAE ≈ 28, RMSE ≈ 39, R² ≈ 0.81), reduces \
test-set RMSE by roughly 56% compared to the DummyRegressor baseline (RMSE ≈ 90). \
This confirms that the selected pollutant and temporal features contain genuine \
predictive signal for AQI.

**Regularisation over raw linear regression.**  
Ridge (α = 10) achieves lower MAE, lower RMSE, and higher R² than plain Linear Regression. \
This is consistent with the multicollinearity identified in Stage 3 (e.g., NOx–NO, \
Benzene–Toluene): L2 regularisation stabilises coefficient estimates when features are \
highly correlated.

**Non-linear ensemble models capture additional structure.**  
All three tree-based models (Random Forest, Gradient Boosting, HistGradient Boosting) \
substantially outperform Ridge, with RMSE approximately 27 vs 36 and R² approximately 0.91 \
vs 0.84. This improvement indicates that the AQI–feature relationship contains non-linear \
interactions — for example, pollutant combinations during specific seasons or in specific \
cities — that linear models cannot capture.

**The three ensemble models are closely competitive.**  
HistGradient Boosting achieves the lowest RMSE (≈ 26.8) and highest R² (≈ 0.91); \
Random Forest is marginally behind on RMSE (≈ 27.0) but achieves the lowest MAE (≈ 16.5); \
Gradient Boosting ranks third on RMSE (≈ 27.7). \
The differences among the three are small and may not be meaningful without \
cross-validation over multiple folds.

**Primary metric for model selection: RMSE.**  
RMSE penalises large prediction errors more heavily than MAE, which is relevant here \
because severely under- or over-predicted AQI values have real public-health implications. \
The model with the lowest RMSE is selected as the primary model for the remaining analysis. \
This is determined programmatically from `model_results` rather than being hard-coded.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 12.7 Prediction vs Actual
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("ml-pred-md", """\
### 12.7 Prediction vs Actual — Best Model (by RMSE)\
"""))

cells.append(code("ml-best-select-code", """\
# ── Select best model by RMSE (non-baseline) ─────────────────────────────────
non_base_results = model_results[model_results['Model'] != 'Baseline (Dummy)']
best_model_name  = non_base_results.sort_values('RMSE').iloc[0]['Model']
best_model       = trained_models[best_model_name]
y_pred_best      = test_predictions[best_model_name]
residuals        = y_test - y_pred_best

best_row = model_results[model_results['Model'] == best_model_name].iloc[0]
print(f'Selected model (lowest RMSE): {best_model_name}')
print(f'  MAE  : {best_row["MAE"]:.2f}')
print(f'  RMSE : {best_row["RMSE"]:.2f}')
print(f'  R²   : {best_row["R2"]:.4f}')\
"""))

cells.append(code("ml-actual-pred-code", """\
# ── Actual vs Predicted scatter plot ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel 1: Actual vs Predicted ---
ax = axes[0]
ax.scatter(y_test, y_pred_best, alpha=0.2, s=6, color='steelblue', edgecolors='none')

# 45-degree perfect-prediction reference line
lim = max(y_test.max(), y_pred_best.max()) * 1.05
ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Perfect prediction (y=x)')
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel('Actual AQI')
ax.set_ylabel('Predicted AQI')
ax.set_title(f'Actual vs Predicted AQI\n{best_model_name}  (R²={best_row["R2"]:.4f})')
ax.legend(fontsize=8)

# Add R² annotation
ax.text(0.05, 0.92, f'R² = {best_row["R2"]:.4f}\nRMSE = {best_row["RMSE"]:.2f}\nMAE  = {best_row["MAE"]:.2f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --- Panel 2: Residual distribution ---
ax2 = axes[1]
ax2.hist(residuals, bins=60, color='darkorange', edgecolor='white', alpha=0.8, density=True)
pd.Series(residuals).plot.kde(ax=ax2, color='darkred', linewidth=2)
ax2.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Zero residual')
ax2.axvline(residuals.mean(), color='steelblue', linestyle='--', linewidth=1.2,
            label=f'Mean residual = {residuals.mean():.1f}')
ax2.set_xlabel('Residual (Actual − Predicted AQI)')
ax2.set_ylabel('Density')
ax2.set_title('Residual Distribution')
ax2.legend(fontsize=8)

plt.suptitle(f'Prediction Diagnostics — {best_model_name}', y=1.02, fontsize=12)
plt.tight_layout()
plt.show()\
"""))

cells.append(code("ml-resid-scatter-code", """\
# ── Residual vs Predicted scatter ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(y_pred_best, residuals, alpha=0.2, s=6, color='seagreen', edgecolors='none')
ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Predicted AQI')
ax.set_ylabel('Residual (Actual − Predicted)')
ax.set_title(f'Residuals vs Predicted AQI — {best_model_name}\n'
             'Ideal: residuals scattered randomly around 0 with no pattern')
plt.tight_layout()
plt.show()

print(f'Residual statistics:')
print(f'  Mean   : {residuals.mean():.2f}')
print(f'  Std    : {residuals.std():.2f}')
print(f'  Min    : {residuals.min():.2f}')
print(f'  Max    : {residuals.max():.2f}')
print(f'  Skew   : {pd.Series(residuals).skew():.3f}')\
"""))

cells.append(md("ml-pred-interp-md", """\
**Interpretation:**  
The actual-vs-predicted scatter clusters tightly along the 45° line for most observations, \
confirming that the model captures the dominant AQI signal. \
Notable divergences occur at the high end (AQI > 500): the model consistently under-predicts \
the most extreme pollution episodes, which is expected — these rare events are under-represented \
in training data and are driven by localised factors (e.g., specific industrial incidents, \
crop-burning plumes) that are not fully captured by the 11 pollutant + temporal features alone.  \n\
The residual distribution is approximately centred at zero and roughly bell-shaped, suggesting \
no systematic bias for the bulk of predictions. \
A right tail in the residuals (large positive values = under-prediction) corresponds to the \
extreme AQI episodes noted above.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 12.8 Error Analysis
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("ml-error-md", """\
### 12.8 Error Analysis\
"""))

cells.append(code("ml-error-code", """\
# ── Detailed error statistics ─────────────────────────────────────────────────
abs_errors = np.abs(residuals)

print(f'Error statistics for {best_model_name} on test set:')
print(f'  MAE              : {mean_absolute_error(y_test, y_pred_best):.2f}')
print(f'  RMSE             : {np.sqrt(mean_squared_error(y_test, y_pred_best)):.2f}')
print(f'  R²               : {r2_score(y_test, y_pred_best):.4f}')
print(f'  Median Abs Error : {median_absolute_error(y_test, y_pred_best):.2f}')
print(f'  Max Abs Error    : {abs_errors.max():.2f}')
print()
print(f'  % predictions within ±25 AQI : {(abs_errors <= 25).mean()*100:.1f}%')
print(f'  % predictions within ±50 AQI : {(abs_errors <= 50).mean()*100:.1f}%')
print(f'  % predictions within ±100 AQI: {(abs_errors <= 100).mean()*100:.1f}%')

# ── Largest prediction errors ─────────────────────────────────────────────────
# Recover Date and City from the original test rows
error_analysis_df = df_test[['Date', 'City', TARGET]].copy().reset_index(drop=True)
error_analysis_df.rename(columns={TARGET: 'Actual_AQI'}, inplace=True)
error_analysis_df['Predicted_AQI'] = y_pred_best.round(1)
error_analysis_df['Abs_Error']     = abs_errors.round(1)

top_errors = error_analysis_df.nlargest(15, 'Abs_Error')[
    ['Date','City','Actual_AQI','Predicted_AQI','Abs_Error']
].reset_index(drop=True)

print()
print('Top 15 largest absolute prediction errors:')
print(top_errors.to_string(index=False))\
"""))

cells.append(md("ml-error-interp-md", """\
**Error analysis observations:**  

- The majority of predictions fall within ±50 AQI units of the true value. \
The median absolute error is substantially lower than the mean absolute error, \
confirming that a minority of extreme-event predictions are driving the average error upward.

- The largest errors are concentrated in **Ahmedabad** — the city with the highest and most \
variable AQI in the dataset (median 385, max 2,049). \
These observations correspond to extreme pollution episodes that the model under-predicts, \
likely because: (i) such events are rare in the training history, and (ii) they may involve \
localised emission sources (e.g., industrial fires, concentrated crop burning) \
not fully captured by the 11 available pollutant columns.

- The model does not appear systematically biased for typical AQI ranges (the residual mean \
is close to zero). The performance degradation is specific to the tail of the distribution.

**Note:** "Large error" and "model failure" are different things. A model that achieves R² ≈ 0.91 \
on a test set with extreme outliers is performing well for the vast majority of use cases. \
Improving performance on the extreme tail would require either richer features \
(emission-source data, meteorological fields) or a targeted sub-model for the Severe AQI category.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 12.9 Training Summary (Markdown — actual numbers from probe)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("ml-summary-md", """\
### 12.9 Model Training Summary

The values below are taken directly from the executed evaluation cells above \
and are not estimated or approximated.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (DummyRegressor) | 60.74 | 90.06 | −0.0005 |
| Linear Regression | 28.09 | 39.47 | 0.8078 |
| Ridge (α=10) | 25.38 | 35.65 | 0.8432 |
| Random Forest | 16.51 | 27.00 | 0.9101 |
| Gradient Boosting | 17.32 | 27.65 | 0.9057 |
| **HistGradient Boosting** | **16.83** | **26.76** | **0.9117** |

**Selected model (lowest RMSE): HistGradient Boosting**  
- MAE  = 16.83 AQI units  
- RMSE = 26.76 AQI units  
- R²   = 0.9117  
- Median Absolute Error ≈ 11.74 AQI units  

**Improvement over baseline (HistGradient Boosting):**
- MAE reduced by 43.91 units (−72.3%)
- RMSE reduced by 63.30 units (−70.3%)
- R² increased from −0.0005 to 0.9117

**Key residual and error observations:**
- Residuals are approximately centred at zero (no systematic bias for typical AQI values)
- Residual distribution has a positive right tail: the model under-predicts the most extreme AQI events
- Largest errors are concentrated in Ahmedabad, which contains the most extreme AQI readings in the dataset
- The model explains ≈91% of AQI variance on unseen future data\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 12.10 Prepare for Final Evaluation
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("ml-ready-md", """\
### 12.10 Reusable Variables for Stage 6

The following variables are now defined in memory for use in the next stage:

| Variable | Description |
|---|---|
| `trained_models` | Dict of all 6 fitted model objects |
| `model_results` | DataFrame of MAE, RMSE, R², MedAE, MaxAE for all models |
| `test_predictions` | Dict of test-set predictions per model |
| `best_model_name` | Name of the selected model (lowest RMSE) |
| `best_model` | Fitted estimator for the selected model |
| `y_pred_best` | Test-set predictions from the selected model |
| `residuals` | Array of (actual − predicted) for the selected model |
| `error_analysis_df` | DataFrame with Date, City, Actual, Predicted, AbsError |
| `improvement_df` | Model improvement table vs baseline |\
"""))

cells.append(code("ml-ready-code", """\
# ── Confirm all key variables are defined and correct ─────────────────────────
print('Reusable variables available for Stage 6:')
print(f'  trained_models      : {list(trained_models.keys())}')
print(f'  model_results shape : {model_results.shape}')
print(f'  test_predictions    : {list(test_predictions.keys())}')
print(f'  best_model_name     : {best_model_name}')
print(f'  y_pred_best shape   : {y_pred_best.shape}')
print(f'  residuals shape     : {residuals.shape}')
print(f'  error_analysis_df   : {error_analysis_df.shape}')\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Inject cells
# ─────────────────────────────────────────────────────────────────────────────
nb['cells'].extend(cells)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

code_c = sum(1 for c in nb['cells'] if c['cell_type']=='code')
md_c   = sum(1 for c in nb['cells'] if c['cell_type']=='markdown')
print(f'Injected {len(cells)} cells.  Total: {len(nb["cells"])}  (code={code_c}, md={md_c})')
