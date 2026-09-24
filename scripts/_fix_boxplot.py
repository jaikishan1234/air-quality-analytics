import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix the city boxplot cell
fixes = {
    "eda-city-box-code": """# ── Box plot: AQI distribution across cities ─────────────────────────────────
city_order = city_stats.index.tolist()  # sorted by median AQI

fig, ax = plt.subplots(figsize=(14, 7))
data_by_city = [df_aqi[df_aqi['City'] == city]['AQI'].values for city in city_order]
bp = ax.boxplot(data_by_city, tick_labels=city_order, patch_artist=True,
                medianprops=dict(color='darkred', linewidth=1.5),
                flierprops=dict(marker='o', markersize=2, alpha=0.2, markerfacecolor='grey'),
                boxprops=dict(facecolor='#AEC6CF', color='steelblue'),
                whiskerprops=dict(color='steelblue'),
                capprops=dict(color='steelblue'))
ax.set_xticklabels(city_order, rotation=45, ha='right', fontsize=8.5)
ax.set_ylabel('AQI')
ax.set_title('AQI Distribution by City (sorted by median AQI)')
for val, label, clr in [(100, 'Satisfactory', '#2ca02c'),
                          (200, 'Moderate', '#ff7f0e'),
                          (300, 'Poor', '#d62728')]:
    ax.axhline(val, linestyle='--', linewidth=0.9, color=clr, alpha=0.7, label=label)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()""",
    "eda-seasonal-code": """# ── Season feature + AQI comparison ─────────────────────────────────────────
def assign_season(month):
    if month in [12, 1, 2]:    return 'Winter'
    if month in [3, 4, 5]:     return 'Spring/Pre-Monsoon'
    if month in [6, 7, 8, 9]:  return 'Monsoon'
    return 'Post-Monsoon/Autumn'

df['Season']     = df['Month'].map(assign_season)
df_aqi['Season'] = df_aqi['Month'].map(assign_season)

season_order = ['Winter', 'Spring/Pre-Monsoon', 'Monsoon', 'Post-Monsoon/Autumn']

season_stats = (
    df_aqi.groupby('Season')['AQI']
    .agg(Median='median', Mean='mean', Std='std', Count='count')
    .round(1)
    .reindex(season_order)
)
print('AQI by Season:')
print(season_stats.to_string())

# Box plot by season
fig, ax = plt.subplots(figsize=(10, 5))
season_colors = ['#4e79a7', '#f28e2c', '#59a14f', '#e15759']
data_by_season = [df_aqi[df_aqi['Season'] == s]['AQI'].values for s in season_order]
bp = ax.boxplot(data_by_season, tick_labels=season_order, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markersize=2, alpha=0.2, markerfacecolor='grey'))
for patch, color in zip(bp['boxes'], season_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('AQI')
ax.set_title('AQI Distribution by Season')
for val, label, clr in [(100,'Satisfactory','#2ca02c'),
                          (200,'Moderate','#ff7f0e'),
                          (300,'Poor','#d62728')]:
    ax.axhline(val, linestyle='--', linewidth=0.9, color=clr, alpha=0.6, label=label)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()"
"""
}

for cell in nb['cells']:
    cell_id = cell.get('id', '')
    if cell_id in fixes:
        # Split fix source into list of strings per line
        new_src = fixes[cell_id].strip()
        cell['source'] = [line + '\n' for line in new_src.split('\n')]
        # Remove trailing \n from last element
        if cell['source'] and cell['source'][-1].endswith('\n'):
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
        print(f'Fixed cell: {cell_id}')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Done.')
