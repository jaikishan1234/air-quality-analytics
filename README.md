# Air Quality Analytics and AQI Prediction in India

**IBM SkillsBuild Data Analytics with AI Internship 2026**

A data analytics and machine learning project that analyzes daily air-quality measurements from Indian cities and builds regression models to estimate Air Quality Index (AQI).

## Project Overview

This project uses the **Air Quality Data in India (2015–2020)** dataset from Kaggle. The workflow covers data understanding, data-quality assessment, cleaning, exploratory data analysis, feature engineering, regression modeling, model comparison, and error analysis.

The project treats **AQI as a continuous regression target** and deliberately excludes `AQI_Bucket` from the predictor set because it is derived from AQI.

## Dataset

- **Dataset:** Air Quality Data in India (2015–2020)
- **Source:** Kaggle
- **File used:** `city_day.csv`
- **Original observations:** 29,531
- **Cities:** 26
- **Date range:** 2015-01-01 to 2020-07-01
- **Target:** `AQI`

Dataset page: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

## Workflow

1. Dataset loading and structural inspection
2. Missing-value and data-quality assessment
3. Date conversion and temporal feature creation
4. Removal of rows with no pollutant measurements
5. Removal of `Xylene` because of very high missingness
6. City-aware median imputation for remaining pollutant features
7. Exploratory analysis of AQI, cities, seasons, pollutants, and correlations
8. Feature engineering and city one-hot encoding
9. Chronological 80/20 train-test split
10. Regression model training and comparison
11. Prediction error and permutation-importance analysis
12. Final conclusions, limitations, and future scope

## Data Preparation

After cleaning:

- **28,108** rows remained.
- **1,423** rows with all pollutant measurements missing were removed.
- `Xylene` was dropped because approximately 61% of its values were missing.
- Remaining pollutant missing values were handled using city-aware median imputation with a global-median fallback.
- Rows without a valid AQI were excluded from supervised modeling.
- `AQI_Bucket` was not used as a predictor.

For modeling, **24,801 observations with valid AQI** were used.

## Machine Learning

The project evaluates:

- Dummy Regressor baseline
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor

The modeling workflow uses an **80/20 chronological split**:

- Training: 2015-01-01 to 2019-12-06
- Testing: 2019-12-07 to 2020-07-01

The selected model is the **HistGradientBoosting Regressor**, selected using the lowest test RMSE.

### Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Regressor | 60.74 | 90.06 | -0.0005 |
| Linear Regression | 28.09 | 39.47 | 0.8078 |
| Ridge | 25.38 | 35.65 | 0.8432 |
| Random Forest | 16.51 | 27.00 | 0.9101 |
| Gradient Boosting | 17.32 | 27.65 | 0.9057 |
| HistGradientBoosting | 16.83 | 26.76 | 0.9117 |

The HistGradientBoosting model achieved an RMSE of **26.76 AQI units** and an R² of **0.9117** on the chronological held-out test set.

## Key Findings

- AQI is strongly right-skewed, with median AQI around 118 and extreme observations reaching 2,049.
- City-level air quality varies substantially, with Ahmedabad showing much higher AQI levels than low-AQI cities such as Aizawl.
- Seasonal variation is strong, with lower median AQI during the monsoon and substantially higher levels during winter.
- PM2.5 and CO show strong relationships with AQI.
- Permutation importance identifies **PM2.5, CO, and PM10** as the leading predictive features.
- Model accuracy is much better for typical AQI ranges than for rare extreme pollution events.
- Ahmedabad contains the largest prediction errors, reflecting its extreme AQI observations.

## Limitations

- Meteorological variables and other external environmental factors are not included.
- Some pollutant values required imputation.
- The dataset has unequal representation across cities.
- Extreme AQI events are relatively rare and therefore harder to predict.
- The evaluation period ends in July 2020 and includes the unusual COVID-19 period.
- The analysis is observational and does not establish causal relationships.

## Project Structure

```text
air-quality-analytics/
├── data/
│   └── city_day.csv
├── notebooks/
│   └── JaikishanNayak_AirQualityAnalytics.ipynb
├── .gitignore
├── README.md
├── requirements.txt
└── JaikishanNayak_ProjectReport.docx
```

## How to Run

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/JaikishanNayak_AirQualityAnalytics.ipynb
```

Make sure `data/city_day.csv` exists at the expected relative path before running the notebook.

## Deliverables

- `JaikishanNayak_AirQualityAnalytics.ipynb`
- `requirements.txt`
- `README.md`
- `JaikishanNayak_ProjectReport.docx`

## Author

**Jaikishan Nayak**

IBM SkillsBuild Data Analytics with AI Internship 2026
