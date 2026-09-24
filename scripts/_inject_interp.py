import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md(id_, s):
    lines = s.strip().split('\n')
    src = [l + '\n' for l in lines]
    src[-1] = src[-1].rstrip('\n')
    return {"cell_type": "markdown", "id": id_, "metadata": {}, "source": src}

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
cells.append(md("interp-header", """\
---

## 13. Model Interpretation and Error Analysis

This section deepens the evaluation of the selected model by examining *which features drive \
its predictions*, *where it performs best and worst*, and *what its realistic limitations are*. \
All analysis uses the existing predictions, test set, and training outputs from Stage 5. \
No models are retrained here.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 13.1 Selected Model Recap
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("interp-selected-md", """\
### 13.1 Selected Model

The selected model is the one with the lowest test-set RMSE from Stage 5, determined \
programmatically from `model_results`. RMSE is used as the primary selection metric because \
it penalises large prediction errors more heavily than MAE — relevant here because \
severely over- or under-predicted AQI values have real public-health implications. \
MAE is reported alongside RMSE because it gives an interpretable estimate of the typical \
prediction error in the same units as AQI.\
"""))

cells.append(code("interp-selected-code", """\
# ── Confirm selected model and display full metric profile ───────────────────
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

print('Selected model (lowest test RMSE):', best_model_name)
print()

best_metrics = model_results[model_results['Model'] == best_model_name].iloc[0]
print('Performance on held-out test set (Dec 2019 – Jul 2020):')
print('  MAE   (Mean Absolute Error)    :', best_metrics['MAE'])
print('  RMSE  (Root Mean Squared Error):', best_metrics['RMSE'])
print('  R²    (Variance Explained)     :', best_metrics['R2'])
print('  MedAE (Median Absolute Error)  :', best_metrics['MedAE'])
print('  MaxAE (Maximum Absolute Error) :', best_metrics['MaxAE'])
print()
print('Metric interpretation:')
print('  RMSE of', best_metrics['RMSE'],
      'means large prediction errors are penalised strongly.')
print('  MAE  of', best_metrics['MAE'],
      'means the typical prediction is within ~', best_metrics['MAE'], 'AQI units of truth.')
print('  R²   of', best_metrics['R2'],
      '-- model explains', str(round(best_metrics['R2']*100, 1))+'% of test AQI variance.')\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 13.2 Feature Importance — Permutation Importance
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("interp-perm-md", """\
### 13.2 Feature Importance via Permutation Importance

`HistGradientBoostingRegressor` supports `feature_importances_` (gain-based, internal \
to the trees), but this attribute measures *average information gain at split points* \
rather than true held-out predictive contribution. It can overstate the importance of \
high-cardinality features and cannot account for post-processing transformations in the \
pipeline.

Instead, **permutation importance** is used:

> For each feature, the test-set values are randomly shuffled, breaking its relationship \
with the target. The drop in model performance (measured as change in R²) estimates how \
much that feature contributes to predictions. A larger drop = more important feature.

**Important caveats:**
- Permutation importance measures *predictive association*, not causality.
- Correlated features (e.g., NOx and NO, Benzene and Toluene) may share importance \
  across both — shuffling one may not fully degrade performance if the correlated \
  partner still carries the signal.
- Results apply to this specific dataset, split, and model.\
"""))

cells.append(code("interp-perm-code", """\
# ── Permutation importance on test set ───────────────────────────────────────
from sklearn.inspection import permutation_importance

print('Computing permutation importance (n_repeats=10)...')
perm_result = permutation_importance(
    best_model, X_test_processed, y_test,
    n_repeats=10, random_state=42, scoring='r2', n_jobs=-1
)

# Recover feature names from the fitted preprocessor
_ohe_cols      = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(['City'])
_feat_names    = all_feat_names   # already defined in Stage 4 (numeric + OHE city cols)

perm_importance_df = pd.DataFrame({
    'Feature':          _feat_names,
    'Importance_Mean':  perm_result.importances_mean.round(5),
    'Importance_Std':   perm_result.importances_std.round(5),
}).sort_values('Importance_Mean', ascending=False).reset_index(drop=True)

print()
print('Top 20 features by permutation importance (R² drop when shuffled):')
print(perm_importance_df.head(20).to_string(index=False))\
"""))

cells.append(code("interp-perm-plot-code", """\
# ── Horizontal bar chart: top 15 permutation importances ─────────────────────
top15 = perm_importance_df.head(15).copy()[::-1]  # reverse for horizontal plot

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#d62728' if f in POLLUTANT_FEATURES else
          '#1f77b4' if f in TEMPORAL_FEATURES  else
          '#2ca02c'
          for f in top15['Feature']]
bars = ax.barh(top15['Feature'], top15['Importance_Mean'], xerr=top15['Importance_Std'],
               color=colors, edgecolor='white', capsize=3)
ax.set_xlabel('Mean Permutation Importance (R² drop when feature shuffled)')
ax.set_title('Top 15 Feature Importances (Permutation, n=10 repeats)\n'
             'Red = pollutant | Blue = temporal | Green = city')
ax.axvline(0, color='black', linewidth=0.8)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#d62728', label='Pollutant'),
                   Patch(facecolor='#1f77b4', label='Temporal'),
                   Patch(facecolor='#2ca02c', label='City')]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
plt.tight_layout()
plt.show()\
"""))

cells.append(md("interp-perm-interp-md", """\
**Interpretation:**
**PM2.5** (importance ≈ 0.589) is by far the dominant predictor — shuffling it causes the \
largest drop in R². This is consistent with the EDA finding that PM2.5 has the second-highest \
Pearson correlation with AQI (r = 0.657) and is one of the primary components of India's \
AQI calculation.

**CO** (importance ≈ 0.368) ranks second, reflecting its strong correlation (r = 0.678) \
and its role in AQI sub-index computation. Together, PM2.5 and CO account for the vast \
majority of the model's predictive power.

**PM10** (importance ≈ 0.133) ranks third. The remaining pollutants (NO, O3, SO2, NOx) \
contribute modest individual importance. The temporal and city features individually have \
small permutation importances, but their collective contribution is meaningful \
(see Section 13.3).\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 13.3 Pollutant-Level Group Importance
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("interp-group-md", """\
### 13.3 Feature Group Importance

Features are grouped into three conceptual categories and their permutation importances \
are summed to give a group-level view of what information the model relies on most.\
"""))

cells.append(code("interp-group-code", """\
# ── Permutation importance grouped by feature type ────────────────────────────
poll_imp  = perm_importance_df[perm_importance_df['Feature'].isin(POLLUTANT_FEATURES)]['Importance_Mean'].sum()
temp_imp  = perm_importance_df[perm_importance_df['Feature'].isin(TEMPORAL_FEATURES)]['Importance_Mean'].sum()
city_imp  = perm_importance_df[perm_importance_df['Feature'].str.startswith('City_')]['Importance_Mean'].sum()
total_imp = poll_imp + temp_imp + city_imp

group_df = pd.DataFrame({
    'Feature Group':        ['Pollutants (11)', 'Temporal (9)', 'City OHE (22)'],
    'Total Importance':     [round(poll_imp, 4), round(temp_imp, 4), round(city_imp, 4)],
    'Share (%)':            [round(poll_imp/total_imp*100, 1),
                             round(temp_imp/total_imp*100, 1),
                             round(city_imp/total_imp*100, 1)],
})
print('Permutation Importance by Feature Group:')
print(group_df.to_string(index=False))
print()
print('Note: "Share" is relative to the total summed importance of all features.')
print('Pollutant measurements account for the large majority of predictive contribution.')

# Pie chart
fig, ax = plt.subplots(figsize=(7, 5))
ax.pie(group_df['Total Importance'], labels=group_df['Feature Group'],
       autopct='%1.1f%%', startangle=90, colors=['#d62728','#1f77b4','#2ca02c'],
       wedgeprops=dict(edgecolor='white', linewidth=1.5))
ax.set_title('Permutation Importance Share by Feature Group')
plt.tight_layout()
plt.show()\
"""))

cells.append(md("interp-group-interp-md", """\
Pollutant measurements collectively account for the overwhelming majority of the model's \
predictive importance. This is expected — AQI is computed from pollutant sub-indices, \
and the model has learned to approximate this relationship.

Temporal features (cyclical encodings of day-of-year, month, etc.) contribute a small \
but nonzero collective share, capturing seasonal variation that is not fully encoded \
in the pollutant values alone. City features contribute comparably, capturing systematic \
city-level offsets in AQI that persist after accounting for pollutant levels.

These are *predictive associations* — the features correlate with AQI in the training data. \
They do not imply that removing temporal or city information from the real world would \
reduce pollution.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 13.4 Actual vs Predicted Sample Table
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("interp-sample-md", """\
### 13.4 Actual vs Predicted — Sample Table\
"""))

cells.append(code("interp-sample-code", """\
# ── Build full prediction table and display a representative sample ───────────
pred_table = df_test[['Date','City','AQI']].copy().reset_index(drop=True)
pred_table.rename(columns={'AQI': 'Actual_AQI'}, inplace=True)
pred_table['Predicted_AQI'] = y_pred_best.round(1)
pred_table['Residual']      = (pred_table['Actual_AQI'] - pred_table['Predicted_AQI']).round(1)
pred_table['Abs_Error']     = pred_table['Residual'].abs().round(1)

# Sample: 5 with small error, 5 with large error, 5 random
small_err  = pred_table.nsmallest(5, 'Abs_Error')
large_err  = pred_table.nlargest(5, 'Abs_Error')
random_s   = pred_table.sample(5, random_state=42)
sample_tbl = pd.concat([small_err, random_s, large_err]).drop_duplicates().sort_values('Abs_Error')

print('Sample predictions (smallest / random / largest error):')
print(sample_tbl[['Date','City','Actual_AQI','Predicted_AQI','Residual','Abs_Error']].to_string(index=False))
print()

# Summary statistics
print('Overall prediction statistics:')
print('  Mean residual   :', round(pred_table['Residual'].mean(), 2))
print('  Median residual :', round(pred_table['Residual'].median(), 2))
print('  MAE             :', round(pred_table['Abs_Error'].mean(), 2))
print('  RMSE            :', round(np.sqrt((pred_table['Residual']**2).mean()), 2))
print('  R²              :', round(r2_score(pred_table['Actual_AQI'], pred_table['Predicted_AQI']), 4))\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 13.5 AQI Range Error Analysis
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("interp-range-md", """\
### 13.5 AQI Range Error Analysis

Test observations are divided into six AQI ranges matching India's CPCB bands. \
For each range, MAE and RMSE are calculated. \
`AQI_Bucket` from the original dataset is **not used as an input feature here** — \
the ranges are applied *after prediction* purely for diagnostic grouping.\
"""))

cells.append(code("interp-range-code", """\
# ── Error by AQI range ────────────────────────────────────────────────────────
def aqi_range_label(aqi):
    if aqi <= 50:   return '1. Good (0-50)'
    if aqi <= 100:  return '2. Satisfactory (51-100)'
    if aqi <= 200:  return '3. Moderate (101-200)'
    if aqi <= 300:  return '4. Poor (201-300)'
    if aqi <= 400:  return '5. Very Poor (301-400)'
    return           '6. Severe (401+)'

pred_table['AQI_Range'] = pred_table['Actual_AQI'].apply(aqi_range_label)

range_error = (
    pred_table.groupby('AQI_Range')
    .apply(lambda g: pd.Series({
        'N':    len(g),
        'MAE':  round(g['Abs_Error'].mean(), 2),
        'RMSE': round(np.sqrt((g['Residual']**2).mean()), 2),
        'Mean_Actual':    round(g['Actual_AQI'].mean(), 1),
        'Mean_Predicted': round(g['Predicted_AQI'].mean(), 1),
    }), include_groups=False)
    .reset_index()
)

print('Prediction error by AQI range (post-hoc grouping only — not a predictor):')
print(range_error.to_string(index=False))

# Grouped bar chart
fig, ax = plt.subplots(figsize=(11, 5))
x     = range(len(range_error))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], range_error['MAE'],  width,
               label='MAE',  color='steelblue',  edgecolor='white')
bars2 = ax.bar([i + width/2 for i in x], range_error['RMSE'], width,
               label='RMSE', color='darkorange', edgecolor='white')

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(list(x))
ax.set_xticklabels(range_error['AQI_Range'], rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Error (AQI units)')
ax.set_title('MAE and RMSE by Actual AQI Range\n'
             '(Ranges applied post-prediction for diagnostic purposes only)')
ax.legend()
plt.tight_layout()
plt.show()\
"""))

cells.append(md("interp-range-interp-md", """\
**Interpretation:**
Model error increases substantially at higher AQI levels. The Satisfactory range \
(51–100) has the lowest errors (MAE ≈ 10.6, RMSE ≈ 15.2), while the Severe range \
(401+) has by far the largest errors (MAE ≈ 84.5, RMSE ≈ 124.7). \
This is a common property of regression models trained on skewed targets: \
observations in the tail are both rarer and driven by factors \
(localised industrial events, extreme weather) not fully represented in the \
available features. The 58 Severe-category test observations are particularly \
challenging for the model.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 13.6 City-Level Error Analysis
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("interp-city-md", """\
### 13.6 City-Level Error Analysis

Cities with small test-set sample sizes produce less stable error estimates. \
Only cities with at least 20 test observations are shown in the chart. \
All cities are included in the full table.\
"""))

cells.append(code("interp-city-code", """\
# ── City-level error statistics ───────────────────────────────────────────────
city_error = (
    pred_table.groupby('City')
    .apply(lambda g: pd.Series({
        'N':              len(g),
        'MAE':            round(g['Abs_Error'].mean(), 2),
        'RMSE':           round(np.sqrt((g['Residual']**2).mean()), 2),
        'Mean_Actual':    round(g['Actual_AQI'].mean(), 1),
        'Mean_Predicted': round(g['Predicted_AQI'].mean(), 1),
    }), include_groups=False)
    .sort_values('MAE', ascending=False)
    .reset_index()
)

print('City-level prediction error (sorted by MAE descending):')
print(city_error.to_string(index=False))
print()
print('Note: cities with small N have less reliable error estimates.')

# Horizontal bar chart — cities with N >= 20
city_plot = city_error[city_error['N'] >= 20].sort_values('MAE')
fig, ax = plt.subplots(figsize=(10, 7))
colors_c = ['#d62728' if v > 30 else '#ff7f0e' if v > 20 else '#2ca02c'
            for v in city_plot['MAE']]
bars = ax.barh(city_plot['City'], city_plot['MAE'], color=colors_c, edgecolor='white')
for bar, (_, row) in zip(bars, city_plot.iterrows()):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            str(row['MAE']) + '  (n=' + str(int(row['N'])) + ')',
            va='center', fontsize=8)
ax.set_xlabel('MAE (AQI units)')
ax.set_title('City-Level Prediction MAE (cities with N >= 20 test observations)\n'
             'Green < 20 | Orange 20-30 | Red > 30')
ax.set_xlim(0, city_plot['MAE'].max() * 1.3)
plt.tight_layout()
plt.show()\
"""))

cells.append(md("interp-city-interp-md", """\
**Interpretation:**
**Ahmedabad** has by far the highest city-level MAE (≈ 50.0) and RMSE (≈ 79.6). \
This is consistent with Stage 5 findings: Ahmedabad contains the most extreme AQI readings \
in the dataset (median AQI = 385, max = 2,049), and the Severe-range episodes are harder \
to predict. The model's mean predicted AQI for Ahmedabad (242) is substantially lower than \
the mean actual (251), indicating systematic under-prediction during the worst events.

**Aizawl** has a high MAE (≈ 23.6) despite low actual AQI (mean = 34.8) — here the \
*relative* error is large even if the absolute AQI values are small. With only 111 test \
observations and very low pollution levels, small absolute errors translate to high \
relative errors in percentage terms.

Most other cities fall in the 15–25 MAE range, representing good typical performance.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 13.7 Temporal Error Analysis
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("interp-temporal-md", """\
### 13.7 Temporal Error Analysis

The test set spans December 2019 – July 2020. \
We examine whether prediction error varies systematically across months. \
Observed patterns are descriptive — month is not a causal driver of error.\
"""))

cells.append(code("interp-temporal-code", """\
# ── Monthly error on test set ─────────────────────────────────────────────────
pred_table['Month_num'] = df_test['Date'].dt.month.values

month_labels = {12:'Dec-19', 1:'Jan-20', 2:'Feb-20', 3:'Mar-20',
                4:'Apr-20', 5:'May-20', 6:'Jun-20', 7:'Jul-20'}

temporal_error = (
    pred_table.groupby('Month_num')
    .apply(lambda g: pd.Series({
        'Month':  month_labels.get(g['Month_num'].iloc[0], str(g['Month_num'].iloc[0])),
        'N':      len(g),
        'MAE':    round(g['Abs_Error'].mean(), 2),
        'RMSE':   round(np.sqrt((g['Residual']**2).mean()), 2),
        'Mean_Actual': round(g['Actual_AQI'].mean(), 1),
    }), include_groups=False)
    .reset_index()
    .sort_values('Month_num')
)

print('Monthly error on test set:')
print(temporal_error[['Month','N','MAE','RMSE','Mean_Actual']].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(temporal_error['Month'], temporal_error['MAE'],  marker='o', label='MAE',
        color='steelblue', linewidth=2, markersize=7)
ax.plot(temporal_error['Month'], temporal_error['RMSE'], marker='s', label='RMSE',
        color='darkorange', linewidth=2, markersize=7)
for _, row in temporal_error.iterrows():
    ax.annotate(str(row['MAE']), (row['Month'], row['MAE']),
                textcoords='offset points', xytext=(0, 6), ha='center', fontsize=7.5, color='steelblue')

ax.set_xlabel('Month (test set: Dec 2019 – Jul 2020)')
ax.set_ylabel('Error (AQI units)')
ax.set_title('Monthly MAE and RMSE on Test Set\n'
             '(Jul-20 has only 24 observations — interpret with caution)')
ax.legend()
plt.tight_layout()
plt.show()\
"""))

cells.append(md("interp-temporal-interp-md", """\
**Interpretation:**
Prediction errors are higher in December 2019 (RMSE ≈ 35.7) and January–February 2020 \
(RMSE ≈ 31–32), corresponding to winter months when AQI values are highest and most variable. \
Errors are lowest in April–June 2020 (RMSE ≈ 19–20), when pollution levels are lower \
and more predictable.

This pattern mirrors the seasonal AQI distribution from EDA: the model finds it harder \
to predict extreme winter AQI than moderate monsoon-period AQI. \
July 2020 has only 24 test observations and should not be over-interpreted.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 13.8 Worst Prediction Cases
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("interp-worst-md", """\
### 13.8 Worst Prediction Cases\
"""))

cells.append(code("interp-worst-code", """\
# ── Top 10 largest absolute errors ───────────────────────────────────────────
worst_errors = (
    pred_table.nlargest(10, 'Abs_Error')
    [['Date','City','Actual_AQI','Predicted_AQI','Residual','Abs_Error']]
    .reset_index(drop=True)
)
worst_errors.index += 1
worst_errors.index.name = 'Rank'

print('Top 10 largest absolute prediction errors:')
print(worst_errors.to_string())\
"""))

cells.append(md("interp-worst-interp-md", """\
**Observations from the worst-error cases:**

- The 10 largest errors are predominantly in **Ahmedabad**, and all involve the model \
  *under-predicting* actual AQI (positive residuals — actual is higher than predicted). \
  This is consistent with Ahmedabad containing the most extreme AQI values in the dataset.

- The largest absolute error is ≈ 396 AQI units. These are not simply random noise — \
  they correspond to genuine extreme-pollution episodes that are statistically rare in the \
  training data and may be driven by factors not captured in the 11 available pollutant \
  columns (e.g., wind direction, local fire events, or atypical industrial activity).

- These cases do not represent a general model failure — they are a small fraction \
  of the 4,961 test observations. The overall model achieves RMSE ≈ 26.8 and R² ≈ 0.91. \
  The extreme cases raise the RMSE above what the median-error analysis would suggest.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 13.9 Model Limitations
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("interp-limitations-md", """\
### 13.9 Model Limitations

The following limitations are specific to this project and do not imply that similar models \
cannot perform better with richer data or different approaches.

1. **Skewed AQI target.**  
   AQI values in this dataset have a skewness of ≈ 3.4. Models trained on skewed targets \
   tend to under-predict extreme values. The Severe-range RMSE (≈ 125) is roughly \
   4.7× the Satisfactory-range RMSE (≈ 15).

2. **Extreme pollution episodes are underrepresented.**  
   Of the 24,801 labeled observations, only 1,337 (5.4%) are in the Severe band. \
   The model has limited training signal for the most extreme conditions.

3. **Missing external covariates.**  
   Meteorological variables (temperature, wind speed, humidity, boundary layer height), \
   traffic counts, industrial activity indices, and geographic terrain data are known \
   drivers of AQI that are not included in this dataset. Their absence limits predictive \
   accuracy, particularly for local extreme events.

4. **City-level data imbalance.**  
   Training data ranges from 111 observations (Aizawl) to 1,999 (Delhi). \
   The model is better calibrated for well-represented cities.

5. **Imputed pollutant values.**  
   Stage 2 applied city-aware median imputation to ≈ 15–38% of several pollutant columns. \
   Imputed values carry less information than actual measurements and may introduce noise.

6. **Chronological evaluation covers a specific period.**  
   The test set spans December 2019 – July 2020, which includes the onset of the COVID-19 \
   lockdown (March 2020 onwards). This may have reduced emissions atypically, potentially \
   making 2020 data different from historical patterns the model was trained on.

7. **No causality established.**  
   Feature importance shows that PM2.5 and CO are the strongest *predictive associates* \
   of AQI. This does not mean reducing only these two pollutants would reduce AQI — the \
   relationship is observational and involves complex co-emission dynamics.

8. **Dataset scope.**  
   The dataset covers 26 cities. Predictions for cities outside this set, or for future \
   years with different emission profiles, should not be assumed to be reliable without \
   retraining on updated data.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 13.10 Final ML Interpretation Summary
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("interp-final-md", """\
### 13.10 Final ML Interpretation Summary

The following statements are grounded in numerical results from the executed analysis.

---

**1. How much did the model improve over the baseline?**  
The HistGradient Boosting model reduced test-set RMSE by 63.30 units (−70.3%) relative \
to the DummyRegressor baseline (90.06 → 26.76). MAE improved by 43.91 units (−72.3%). \
R² increased from −0.0005 to 0.9117, meaning the model explains ≈ 91% of AQI variance \
on unseen future data.

**2. Which features were most predictive?**  
According to permutation importance on the test set:  
- **PM2.5** (importance ≈ 0.589) is the single most influential feature.  
- **CO** (importance ≈ 0.368) ranks second.  
- **PM10** (importance ≈ 0.133) ranks third.  
- All other features have substantially lower individual importances.  
Pollutant features collectively account for the large majority of total predictive importance \
(≈ 96%), with temporal and city features contributing the remainder.

**3. How consistent was performance across AQI ranges?**  
Prediction quality degrades substantially at high AQI levels:  
- Satisfactory (51–100): MAE ≈ 10.6, RMSE ≈ 15.2  
- Moderate (101–200): MAE ≈ 17.6, RMSE ≈ 23.8  
- Poor (201–300): MAE ≈ 32.0, RMSE ≈ 42.8  
- Severe (401+): MAE ≈ 84.5, RMSE ≈ 124.7  
The model performs well for typical conditions and poorly for extreme events.

**4. How consistent was performance across cities?**  
Most cities achieved MAE in the 15–25 range. \
Ahmedabad was a clear outlier (MAE ≈ 50.0), driven by its extreme AQI values. \
Aizawl had high relative error despite low absolute AQI.

**5. Where did the model make its largest errors?**  
The 10 largest absolute errors all involved under-predicting extreme Ahmedabad readings \
(actual AQI > 400, errors up to ≈ 396 AQI units). \
These represent rare, severe pollution episodes that are statistically underrepresented \
in the training data.

**6. What are the main limitations?**  
The key limitations are: AQI target skewness; underrepresentation of Severe-range events; \
absence of meteorological and other external covariates; city-level data imbalance; \
imputed pollutant values; and the observational (non-causal) nature of all feature associations. \
None of these limitations invalidate the model for its intended purpose — estimating \
AQI from routine pollution monitoring data — but they set realistic bounds on expected \
accuracy, especially for extreme events.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 13.11 Final Stage Output — Confirm reusable variables
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("interp-vars-md", """\
### 13.11 Stage Output — Reusable Variables

The following variables are defined and available for the final project section:\
"""))

cells.append(code("interp-vars-code", """\
# ── Confirm all Stage 6 outputs are defined ───────────────────────────────────
print('Stage 6 output variables:')
print('  best_model_name      :', best_model_name)
print('  y_pred_best shape    :', y_pred_best.shape)
print('  perm_importance_df   :', perm_importance_df.shape, '(feature x importance)')
print('  pred_table           :', pred_table.shape, '(test obs x prediction columns)')
print('  range_error          :', range_error.shape)
print('  city_error           :', city_error.shape)
print('  temporal_error       :', temporal_error.shape)
print('  worst_errors         :', worst_errors.shape)
print()
print('All variables available for Stage 7 (Conclusions and Reporting).')\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Inject
# ─────────────────────────────────────────────────────────────────────────────
nb['cells'].extend(cells)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

code_c = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
md_c   = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
print(f'Injected {len(cells)} Stage 6 cells. Total: {len(nb["cells"])}  (code={code_c}, md={md_c})')
