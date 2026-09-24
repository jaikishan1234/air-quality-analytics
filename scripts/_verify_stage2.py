import pandas as pd

RAW_DATA_PATH = 'data/city_day.csv'
POLLUTANT_COLS_12 = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                     'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

df_raw = pd.read_csv(RAW_DATA_PATH)

# 1. Original shape
orig_rows, orig_cols = df_raw.shape
print("=== 1. Original shape ===")
print(f"Rows: {orig_rows:,}  Columns: {orig_cols}")

# 2. All-pollutant-missing rows
all_missing_mask = df_raw[POLLUTANT_COLS_12].isnull().all(axis=1)
rows_all_missing = int(all_missing_mask.sum())
print("\n=== 2. Rows where ALL 12 pollutant columns are NaN ===")
print(f"Count: {rows_all_missing:,}")

# 3. Original missing AQI (raw)
orig_aqi_missing = int(df_raw['AQI'].isnull().sum())
print("\n=== 3. Missing AQI in raw dataframe ===")
print(f"Missing AQI (raw): {orig_aqi_missing:,}")

# 4. Missing AQI after removing all-pollutant-missing rows
df_after_remove = df_raw[~all_missing_mask].copy()
after_aqi_missing = int(df_after_remove['AQI'].isnull().sum())
print("\n=== 4. Missing AQI after removing all-pollutant-missing rows ===")
print(f"Missing AQI (after removal): {after_aqi_missing:,}")

# 5. Difference explanation
diff = orig_aqi_missing - after_aqi_missing
print("\n=== 5. Difference in missing-AQI count ===")
print(f"Original missing AQI : {orig_aqi_missing:,}")
print(f"After removal        : {after_aqi_missing:,}")
print(f"Difference           : {diff:,}")
# How many removed rows also had AQI missing?
removed_rows_aqi_missing = int(df_raw[all_missing_mask]['AQI'].isnull().sum())
print(f"Removed rows that also had AQI missing: {removed_rows_aqi_missing:,}")
print(f"Check: {orig_aqi_missing} - {removed_rows_aqi_missing} = {orig_aqi_missing - removed_rows_aqi_missing}")

# 6. Consistency check against reported numbers
reported_orig    = 29531
reported_removed = 1423
reported_final   = 28108
reported_valid_aqi = 24801
reported_missing_aqi = 3307

print("\n=== 6. Consistency check against reported numbers ===")
print(f"Reported original rows : {reported_orig:,}  | Actual: {orig_rows:,}  | Match: {orig_rows == reported_orig}")
print(f"Reported rows removed  : {reported_removed:,}  | Actual: {rows_all_missing:,}  | Match: {rows_all_missing == reported_removed}")
print(f"Reported final rows    : {reported_final:,} | Actual: {orig_rows - rows_all_missing:,} | Match: {orig_rows - rows_all_missing == reported_final}")

# For valid AQI & missing AQI: these come AFTER Xylene drop (Xylene drop doesn't change row count)
# But they also come AFTER Xylene is dropped from POLLUTANT_COLS — Xylene drop doesn't affect AQI
# Actual final valid/missing AQI is from df_after_remove
actual_valid_aqi   = int(df_after_remove['AQI'].notna().sum())
actual_missing_aqi = int(df_after_remove['AQI'].isnull().sum())
print(f"Reported valid AQI     : {reported_valid_aqi:,} | Actual: {actual_valid_aqi:,}  | Match: {actual_valid_aqi == reported_valid_aqi}")
print(f"Reported missing AQI   : {reported_missing_aqi:,}  | Actual: {actual_missing_aqi:,}  | Match: {actual_missing_aqi == reported_missing_aqi}")
print(f"valid + missing = total rows: {actual_valid_aqi} + {actual_missing_aqi} = {actual_valid_aqi + actual_missing_aqi} | expected {orig_rows - rows_all_missing}")

# 7. If inconsistency found, explain
print("\n=== 7. Inconsistency analysis ===")
consistent = (
    orig_rows == reported_orig and
    rows_all_missing == reported_removed and
    orig_rows - rows_all_missing == reported_final and
    actual_valid_aqi == reported_valid_aqi and
    actual_missing_aqi == reported_missing_aqi
)
if consistent:
    print("All reported numbers are internally consistent. No discrepancies found.")
else:
    if actual_valid_aqi != reported_valid_aqi:
        print(f"DISCREPANCY in valid AQI: reported {reported_valid_aqi}, actual {actual_valid_aqi} (diff={reported_valid_aqi - actual_valid_aqi})")
    if actual_missing_aqi != reported_missing_aqi:
        print(f"DISCREPANCY in missing AQI: reported {reported_missing_aqi}, actual {actual_missing_aqi} (diff={reported_missing_aqi - actual_missing_aqi})")
    # Detailed breakdown
    print(f"\nBreakdown of the {rows_all_missing} removed rows:")
    removed = df_raw[all_missing_mask]
    print(f"  AQI present in removed rows : {removed['AQI'].notna().sum()}")
    print(f"  AQI missing in removed rows : {removed['AQI'].isnull().sum()}")

print()
