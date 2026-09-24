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
# Section 14 — Final Conclusion
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("concl-header", """\
---

## 14. Final Conclusion

This section synthesises the complete analysis — from raw data ingestion to model \
evaluation — into a cohesive project conclusion. Every numerical claim is drawn from \
results produced earlier in this notebook.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 14.1 Key Findings
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("concl-findings-md", """\
### 14.1 Key Findings

The following findings represent the most important insights produced by this project. \
Each is directly supported by a metric, chart, or table from the preceding analysis.

1. **The AQI distribution is strongly right-skewed (skewness ≈ 3.4).**
   Median AQI across all valid observations is 118 (Moderate band), but the mean is 167, \
   pulled upward by extreme events reaching up to 2,049. Only 5.4% of days are classified \
   as Good — the same proportion as the Severe category.

2. **City-level air quality varies enormously — a 16× difference between extremes.**
   Aizawl (north-east India) has a median AQI of 23 (Good), while Ahmedabad has a median \
   AQI of 385 (Very Poor). Delhi (257), Patna (215), and Gurugram (209) form a \
   high-pollution cluster of major northern cities.

3. **A strong seasonal cycle dominates the temporal pattern.**
   Monsoon-season median AQI (89 — Satisfactory) is approximately half the Winter median \
   (171 — Moderate/Poor). A sharp AQI spike occurs in October–November, consistent with \
   post-harvest crop-residue burning and the onset of stable winter atmospheric conditions.

4. **Year-on-year AQI declined from 2015 to 2019 (175 → 109, −38%).**
   This trend is consistent with improvements in monitoring coverage and possible \
   reductions in industrial emissions. The 2020 figure (93) reflects only January–July \
   and includes the COVID-19 lockdown period, making it difficult to compare with prior years.

5. **PM2.5 and CO are the strongest pollutant predictors of AQI.**
   Pearson correlation with AQI: CO = 0.678, PM2.5 = 0.657. Permutation importance \
   confirms this ordering: PM2.5 (importance ≈ 0.589), CO (≈ 0.368), PM10 (≈ 0.133). \
   These three pollutants together account for almost all of the model's predictive power.

6. **Non-linear ensemble models substantially outperform linear models.**
   HistGradient Boosting achieved RMSE = 26.76 and R² = 0.9117 on the chronological test set, \
   compared to Ridge Regression (RMSE = 35.65, R² = 0.84). This 25% RMSE improvement \
   demonstrates that the AQI–pollutant relationship contains non-linear interactions \
   that linear models cannot capture.

7. **The selected model improved over the trivial baseline by over 70%.**
   The DummyRegressor (predict training median = 124.0 for every observation) achieves \
   RMSE = 90.06 and R² ≈ 0 on the test set. HistGradient Boosting reduced RMSE by \
   63.30 units (−70.3%) and MAE by 43.91 units (−72.3%).

8. **Prediction is accurate for typical AQI ranges but degrades at the extremes.**
   Satisfactory-range (51–100) predictions have MAE ≈ 10.6 and RMSE ≈ 15.2. \
   Severe-range (401+) predictions have MAE ≈ 84.5 and RMSE ≈ 124.7. \
   Overall, 80.3% of test predictions fall within ±25 AQI units of the true value, \
   and 95.5% fall within ±50 units.

9. **Ahmedabad is by far the hardest city to predict accurately.**
   City-level MAE: Ahmedabad ≈ 50.0, versus a typical range of 15–25 for most other cities. \
   All 10 largest absolute prediction errors originate from Ahmedabad, with the largest \
   being 396 AQI units (actual = 988, predicted = 592, March 2020).

10. **Data quality challenges required systematic decisions at every stage.**
    Xylene was dropped due to 61% missingness. 1,423 rows with no pollutant measurements \
    were removed. City-aware median imputation was applied to the remaining missing values \
    in 11 pollutant columns. AQI was never imputed, as it is the prediction target.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 14.2 Project Limitations
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("concl-limitations-md", """\
### 14.2 Project Limitations

1. **Missing external covariates.** Meteorological variables (temperature, wind, humidity, \
   atmospheric boundary layer height), traffic data, industrial activity records, and \
   geographic context are known drivers of AQI that were not available in this dataset. \
   Their absence limits predictive accuracy, especially for localised extreme events.

2. **Imputed pollutant values.** City-aware median imputation was applied to up to 38% \
   of values in some pollutant columns. Imputed values reduce information quality and \
   may introduce systematic bias in cities with high missingness.

3. **Historical evaluation scope.** The chronological test set (December 2019 – July 2020) \
   reflects a specific historical period including the COVID-19 lockdown, which may have \
   produced atypically low emissions. Model performance on other future periods may differ.

4. **Underrepresentation of extreme events.** Only 5.4% of observations are in the Severe \
   AQI category. The model is poorly calibrated for rare, extreme pollution episodes.

5. **Unequal city coverage.** Training data ranges from 111 (Aizawl) to 1,999 (Delhi) \
   observations per city, creating unequal model calibration across the 26 cities.

6. **No causality established.** Feature importance identifies predictive associations, \
   not causal mechanisms. The model does not and cannot determine what interventions \
   would reduce AQI.

7. **Dataset scope and recency.** The dataset covers 26 Indian cities through July 2020. \
   Predictions for cities, time periods, or countries outside this scope should not be \
   assumed to be valid without retraining on relevant data.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 14.3 Future Scope
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("concl-future-md", """\
### 14.3 Future Scope

The following extensions are realistic directions for future work:

1. **Incorporate meteorological features.** Adding temperature, wind speed/direction, \
   humidity, and boundary layer height as covariates would likely improve prediction \
   accuracy, especially for extreme events driven by atmospheric stability.

2. **Incorporate traffic and geographic data.** Vehicle count data, road density maps, \
   and industrial facility locations could provide city-level contextual features that \
   capture local emission sources.

3. **Acquire more recent observations.** Extending the dataset beyond July 2020 would \
   allow evaluation of model generalisability to post-pandemic emission patterns.

4. **Hyperparameter optimisation.** The current HistGradient Boosting model used \
   reasonable-default settings. Systematic tuning via cross-validation (e.g., Optuna, \
   GridSearchCV) could further reduce RMSE.

5. **Targeted modelling for extreme AQI events.** A two-stage approach — first classify \
   whether an observation is Severe, then apply a specialised regressor for Severe cases — \
   could improve accuracy at the tail of the distribution.

6. **Spatial modelling.** Treating the 26 cities as nodes in a geographic network and \
   incorporating spatial autocorrelation could improve predictions for cities with sparse \
   data coverage.

7. **Real-time deployment.** The trained model could be packaged as a REST API or \
   dashboard application to provide near-real-time AQI estimates for cities using \
   daily pollutant measurements as inputs.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 14.4 Final Project Summary Table (code cell — uses actual variables)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("concl-summary-md", """\
### 14.4 Final Project Summary\
"""))

cells.append(code("concl-summary-code", """\
# ── Final project summary — populated from actual notebook variables ──────────
best_m = model_results[model_results['Model'] == best_model_name].iloc[0]

summary_data = {
    'Item':  [
        'Project',
        'Dataset',
        'Total rows analyzed',
        'Cities covered',
        'Target variable',
        'ML task',
        'Train / test methodology',
        'Models evaluated',
        'Selected model',
        'Selected model  MAE',
        'Selected model  RMSE',
        'Selected model  R²',
    ],
    'Value': [
        'Air Quality Analytics and AQI Prediction in India',
        'Air Quality Data in India 2015-2020 (Kaggle, city_day.csv)',
        str(len(df_model)) + ' (of 28,108 cleaned rows with valid AQI)',
        str(df_model['City'].nunique()),
        'AQI (Air Quality Index, continuous numeric)',
        'Supervised regression',
        '80% / 20% chronological split (train ends 2019-12-06, test from 2019-12-07)',
        str(len(model_results)) + ' (DummyRegressor, Linear, Ridge, Random Forest, GBM, HistGBM)',
        best_model_name,
        str(best_m['MAE']) + ' AQI units',
        str(best_m['RMSE']) + ' AQI units',
        str(best_m['R2']),
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 14.5 Closing statement
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("concl-closing-md", """\
### 14.5 Closing Statement

This project demonstrated a complete data-analytics and machine-learning pipeline applied \
to a real, publicly available environmental dataset. Starting from raw daily air-quality \
measurements with substantial missing data and extreme-value distributions, the project \
produced a reproducible analytical workflow covering data cleaning, exploratory analysis, \
feature engineering, model training, and quantitative evaluation.

The selected model — **HistGradient Boosting Regressor** — explains **91.2% of AQI variance** \
on a held-out chronological test set, reducing the median absolute error from 48.0 AQI units \
(trivial baseline) to **11.7 AQI units**. The analysis confirms that PM2.5, CO, and PM10 are \
the dominant predictive features, consistent with both the Pearson correlation analysis in \
Stage 3 and the permutation importance analysis in Stage 6.

The project is academically honest: all limitations are documented, no causal claims are made \
from correlations, and all reported metrics derive from executed code on real data with no \
manual manipulation of the source CSV.\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 14.6 Project Deliverables
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("concl-deliverables-md", """\
---

## Project Deliverables

The following files constitute the complete submission for the \
**IBM SkillsBuild Data Analytics with AI Internship 2026**:

| File | Description |
|---|---|
| `JaikishanNayak_AirQualityAnalytics.ipynb` | Primary project notebook — complete analysis, code, outputs, and visualisations |
| `requirements.txt` | Python package dependencies for reproducing the environment |
| `JaikishanNayak_ProjectReport.docx` | Formatted project report summarising methodology and findings |
| `README.md` | Project overview, setup instructions, and directory structure |

**Notebook structure summary:**

| Section | Content |
|---|---|
| 1–4 | Project Overview, Problem Statement, Objectives, Dataset Description |
| 5–6 | Library Imports, Dataset Loading |
| 7–8 | Data Understanding, Data Quality Assessment |
| 9 | Data Cleaning and Preprocessing |
| 10 | Exploratory Data Analysis |
| 11 | Feature Engineering and ML Dataset Preparation |
| 12 | AQI Prediction Models (Training and Comparison) |
| 13 | Model Interpretation and Error Analysis |
| 14 | Final Conclusion, Limitations, Future Scope |\
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Inject
# ─────────────────────────────────────────────────────────────────────────────
nb['cells'].extend(cells)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

code_c = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
md_c   = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
print(f'Injected {len(cells)} Stage-7 cells.')
print(f'Total: {len(nb["cells"])}  (code={code_c}, md={md_c})')
