import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── helpers ──────────────────────────────────────────────────────────────────
def md(id_, source):
    return {"cell_type": "markdown", "id": id_, "metadata": {}, "source": source}

def code(id_, source):
    return {"cell_type": "code", "execution_count": None, "id": id_,
            "metadata": {}, "outputs": [], "source": source}

# ── STAGE 3 CELLS ─────────────────────────────────────────────────────────────

cells = []

# ── 10. EDA Overview ─────────────────────────────────────────────────────────
cells.append(md("eda-overview-md", [
    "---\n",
    "\n",
    "## 10. Exploratory Data Analysis\n",
    "\n",
    "### 10.1 EDA Overview\n",
    "\n",
    "Exploratory Data Analysis (EDA) is the process of systematically examining the dataset to understand "
    "its structure, distributions, and relationships before drawing conclusions or building models. "
    "At this stage, no assumptions are imposed — every insight is derived from the data itself.\n",
    "\n",
    "**Questions this EDA is designed to answer:**\n",
    "\n",
    "| # | Question |\n",
    "|---|---|\n",
    "| 1 | How is AQI distributed across all observations? |\n",
    "| 2 | How are AQI categories (Good → Severe) distributed? |\n",
    "| 3 | Which cities have the highest and lowest typical AQI? |\n",
    "| 4 | How has overall air quality changed year-on-year from 2015 to 2020? |\n",
    "| 5 | Which months and seasons are associated with the worst air quality? |\n",
    "| 6 | How are individual pollutant concentrations distributed? |\n",
    "| 7 | Which pollutants are most strongly correlated with AQI? |\n",
    "| 8 | How do pollutant profiles differ across cities? |\n",
    "\n",
    "**Variables investigated:**  \n",
    "- **Target:** `AQI` (continuous) and `AQI_Bucket` (descriptive label only)  \n",
    "- **Temporal:** `Year`, `Month`, `Season`  \n",
    "- **Categorical:** `City`  \n",
    "- **Numeric predictors:** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene"
]))

# ── 10.2 AQI Distribution ────────────────────────────────────────────────────
cells.append(md("eda-aqi-dist-md", [
    "### 10.2 AQI Distribution\n",
    "\n",
    "We begin by examining the overall distribution of AQI across all valid observations. "
    "Understanding the shape of this distribution — including skewness, central tendency, and the "
    "presence of extreme values — is essential before building any predictive model."
]))

cells.append(code("eda-aqi-dist-code", [
    "# ── AQI distribution: statistics + histogram + box plot ─────────────────────\n",
    "df_aqi = df.dropna(subset=['AQI']).copy()  # supervised-learning subset (AQI present)\n",
    "\n",
    "# Descriptive statistics\n",
    "aqi_stats = df_aqi['AQI'].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.90, 0.95])\n",
    "aqi_skew  = df_aqi['AQI'].skew()\n",
    "print('AQI Descriptive Statistics')\n",
    "print('─' * 30)\n",
    "for k, v in aqi_stats.items():\n",
    "    print(f'  {k:<8}: {v:,.1f}')\n",
    "print(f'  {\"skewness\":<8}: {aqi_skew:.3f}')\n",
    "\n",
    "# Plot\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Histogram + KDE\n",
    "ax = axes[0]\n",
    "ax.hist(df_aqi['AQI'], bins=60, color='steelblue', edgecolor='white', alpha=0.8, density=True)\n",
    "df_aqi['AQI'].plot.kde(ax=ax, color='darkred', linewidth=2)\n",
    "ax.axvline(df_aqi['AQI'].median(), color='orange', linestyle='--', linewidth=1.5, label=f'Median = {df_aqi[\"AQI\"].median():.0f}')\n",
    "ax.axvline(df_aqi['AQI'].mean(),   color='green',  linestyle='--', linewidth=1.5, label=f'Mean = {df_aqi[\"AQI\"].mean():.0f}')\n",
    "ax.set_xlabel('AQI')\n",
    "ax.set_ylabel('Density')\n",
    "ax.set_title('AQI Distribution (Histogram + KDE)')\n",
    "ax.legend(fontsize=9)\n",
    "\n",
    "# Box plot\n",
    "ax2 = axes[1]\n",
    "ax2.boxplot(df_aqi['AQI'], vert=True, patch_artist=True,\n",
    "            boxprops=dict(facecolor='#AEC6CF', color='#2c5f8a'),\n",
    "            medianprops=dict(color='darkred', linewidth=2),\n",
    "            flierprops=dict(marker='o', markersize=2, alpha=0.3, markerfacecolor='#d62728'),\n",
    "            whiskerprops=dict(color='#2c5f8a'),\n",
    "            capprops=dict(color='#2c5f8a'))\n",
    "ax2.set_ylabel('AQI')\n",
    "ax2.set_xticks([])\n",
    "ax2.set_title('AQI Box Plot')\n",
    "ax2.annotate(f'Median: {df_aqi[\"AQI\"].median():.0f}', xy=(1.02, df_aqi['AQI'].median()),\n",
    "             xycoords=('axes fraction', 'data'), fontsize=9, color='darkred')\n",
    "\n",
    "plt.suptitle('AQI Distribution (n = {:,} observations)'.format(len(df_aqi)), y=1.02, fontsize=12)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("eda-aqi-dist-interp-md", [
    "**Interpretation:**  \n",
    "The AQI distribution is strongly **right-skewed** (skewness ≈ 3.4). "
    "The median AQI is 118 (Moderate band), while the mean is 167 — pulled upward by extreme events. "
    "The 75th percentile sits at 208 (Poor), meaning at least one quarter of all readings exceed the "
    "Poor threshold. The maximum recorded AQI is 2,049, reflecting severe industrial pollution spikes. "
    "This skewness has important implications for modelling: tree-based or robust regression methods "
    "will be more appropriate than linear models applied to raw AQI values."
]))

# ── 10.3 AQI Category Distribution ───────────────────────────────────────────
cells.append(md("eda-aqi-cat-md", [
    "### 10.3 AQI Category Distribution\n",
    "\n",
    "`AQI_Bucket` divides AQI values into six ordinal categories defined by India's Central Pollution "
    "Control Board (CPCB): Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), "
    "Very Poor (301–400), and Severe (401+). "
    "This column is used **for descriptive analysis only** — it will not be used as a predictor "
    "variable in any machine-learning model, since it is directly derived from the AQI target."
]))

cells.append(code("eda-aqi-cat-code", [
    "# ── AQI category distribution ────────────────────────────────────────────────\n",
    "cat_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']\n",
    "cat_counts = df_aqi['AQI_Bucket'].value_counts().reindex(cat_order)\n",
    "cat_pct    = (cat_counts / cat_counts.sum() * 100).round(1)\n",
    "\n",
    "# Summary table\n",
    "cat_summary = pd.DataFrame({'Count': cat_counts, 'Percentage (%)': cat_pct})\n",
    "cat_summary['AQI Range'] = ['0–50', '51–100', '101–200', '201–300', '301–400', '401+']\n",
    "print('AQI Category Distribution')\n",
    "print(cat_summary.to_string())\n",
    "print()\n",
    "\n",
    "# Colour palette: green → dark red\n",
    "cat_colors = ['#2ca02c', '#98df8a', '#ffbb78', '#ff7f0e', '#d62728', '#8c0000']\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Bar chart\n",
    "ax = axes[0]\n",
    "bars = ax.bar(cat_order, cat_counts.values, color=cat_colors, edgecolor='white')\n",
    "for bar, cnt, pct in zip(bars, cat_counts.values, cat_pct.values):\n",
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,\n",
    "            f'{cnt:,}\\n({pct}%)', ha='center', va='bottom', fontsize=8.5)\n",
    "ax.set_xlabel('AQI Category')\n",
    "ax.set_ylabel('Number of Observations')\n",
    "ax.set_title('AQI Category Counts')\n",
    "ax.set_ylim(0, cat_counts.max() * 1.2)\n",
    "\n",
    "# Pie chart\n",
    "ax2 = axes[1]\n",
    "wedges, texts, autotexts = ax2.pie(\n",
    "    cat_counts.values, labels=cat_order, colors=cat_colors,\n",
    "    autopct='%1.1f%%', startangle=90,\n",
    "    wedgeprops=dict(edgecolor='white', linewidth=1.5))\n",
    "for at in autotexts:\n",
    "    at.set_fontsize(8)\n",
    "ax2.set_title('AQI Category Share')\n",
    "\n",
    "plt.suptitle('AQI Category Distribution (descriptive only — not used as ML predictor)',\n",
    "             y=1.02, fontsize=11)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("eda-aqi-cat-interp-md", [
    "**Interpretation:**  \n",
    "The majority of observations fall in the **Moderate** (35.5%) and **Satisfactory** (33.1%) "
    "bands, together accounting for 68.6% of all days with a recorded AQI. "
    "Days classified as Good represent only 5.4% of the dataset — indicating that clean air is "
    "comparatively rare across the monitored cities. "
    "The Severe category (5.4%, 1,337 days) is as frequent as the Good category, "
    "reflecting the severity of pollution in the worst-affected urban centres."
]))

# ── 10.4 City-Level AQI Analysis ─────────────────────────────────────────────
cells.append(md("eda-city-aqi-md", [
    "### 10.4 City-Level AQI Analysis\n",
    "\n",
    "Cities vary considerably in both the level and variability of their AQI. "
    "Because different cities have different numbers of observations (ranging from 111 to 1,999), "
    "comparisons are made on the **median** AQI rather than the mean, which is more robust to the "
    "extreme values present in this dataset. Observation counts are shown alongside each city."
]))

cells.append(code("eda-city-aqi-code", [
    "# ── City-level AQI statistics ────────────────────────────────────────────────\n",
    "city_stats = (\n",
    "    df_aqi.groupby('City')['AQI']\n",
    "    .agg(Median='median', Mean='mean', Min='min', Max='max', Count='count')\n",
    "    .round(1)\n",
    "    .sort_values('Median')\n",
    ")\n",
    "print('City-Level AQI Summary (sorted by median AQI):')\n",
    "print(city_stats.to_string())"
]))

cells.append(code("eda-city-bar-code", [
    "# ── Horizontal bar chart: median AQI by city ─────────────────────────────────\n",
    "fig, ax = plt.subplots(figsize=(10, 9))\n",
    "\n",
    "colors = ['#d62728' if m > 300 else '#ff7f0e' if m > 200\n",
    "          else '#ffbb78' if m > 100 else '#98df8a'\n",
    "          for m in city_stats['Median']]\n",
    "\n",
    "bars = ax.barh(city_stats.index, city_stats['Median'], color=colors, edgecolor='white')\n",
    "for bar, (_, row) in zip(bars, city_stats.iterrows()):\n",
    "    ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height()/2,\n",
    "            f'{row[\"Median\"]:.0f}  (n={row[\"Count\"]:.0f})',\n",
    "            va='center', fontsize=8)\n",
    "\n",
    "# AQI band reference lines\n",
    "for val, label, clr in [(100, 'Satisfactory', '#2ca02c'), (200, 'Moderate', '#ff7f0e'),\n",
    "                          (300, 'Poor', '#d62728')]:\n",
    "    ax.axvline(val, linestyle='--', linewidth=0.9, color=clr, alpha=0.7, label=label)\n",
    "\n",
    "ax.set_xlabel('Median AQI')\n",
    "ax.set_title('Median AQI by City (2015–2020)\\n'\n",
    "             'Colour: green = Satisfactory | orange = Moderate/Poor | red = Very Poor/Severe')\n",
    "ax.legend(fontsize=8, loc='lower right')\n",
    "ax.set_xlim(0, city_stats['Median'].max() * 1.25)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(code("eda-city-box-code", [
    "# ── Box plot: AQI distribution across cities ─────────────────────────────────\n",
    "city_order = city_stats.index.tolist()  # sorted by median AQI\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(14, 7))\n",
    "data_by_city = [df_aqi[df_aqi['City'] == city]['AQI'].values for city in city_order]\n",
    "bp = ax.boxplot(data_by_city, labels=city_order, patch_artist=True,\n",
    "                medianprops=dict(color='darkred', linewidth=1.5),\n",
    "                flierprops=dict(marker='o', markersize=2, alpha=0.2, markerfacecolor='grey'),\n",
    "                boxprops=dict(facecolor='#AEC6CF', color='steelblue'),\n",
    "                whiskerprops=dict(color='steelblue'),\n",
    "                capprops=dict(color='steelblue'))\n",
    "\n",
    "ax.set_xticklabels(city_order, rotation=45, ha='right', fontsize=8.5)\n",
    "ax.set_ylabel('AQI')\n",
    "ax.set_title('AQI Distribution by City (sorted by median AQI)')\n",
    "for val, label, clr in [(100, 'Satisfactory', '#2ca02c'),\n",
    "                          (200, 'Moderate', '#ff7f0e'),\n",
    "                          (300, 'Poor', '#d62728')]:\n",
    "    ax.axhline(val, linestyle='--', linewidth=0.9, color=clr, alpha=0.7, label=label)\n",
    "ax.legend(fontsize=8)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("eda-city-interp-md", [
    "**Interpretation:**  \n",
    "There is a very large range in city-level air quality. "
    "**Aizawl** (Mizoram, north-east India) records the lowest median AQI (23 — Good) with only "
    "111 valid observations, while **Ahmedabad** has the highest median (385 — Very Poor) and the "
    "widest spread, with a maximum of 2,049. "
    "**Delhi** (median 257 — Poor) and **Patna** (215 — Poor) are consistently among the most "
    "polluted large cities. Southern coastal cities such as Thiruvananthapuram, Coimbatore, and "
    "Bengaluru sit in the Satisfactory band.  \n",
    "\n",
    "**Caveat:** Cities with fewer observations (e.g., Aizawl with 111, Shillong with 205) may not "
    "fully capture seasonal peaks, so their medians should be interpreted cautiously."
]))

# ── 10.5 Temporal Analysis ────────────────────────────────────────────────────
cells.append(md("eda-temporal-md", [
    "### 10.5 Temporal Analysis\n",
    "\n",
    "We investigate how AQI has changed year-on-year and whether systematic monthly/seasonal "
    "patterns are present. Median AQI is used to reduce the influence of extreme episodes."
]))

cells.append(code("eda-temporal-code", [
    "# ── Yearly and monthly AQI trends ────────────────────────────────────────────\n",
    "yearly_aqi  = df_aqi.groupby('Year')['AQI'].agg(['median', 'mean', 'count']).round(1)\n",
    "monthly_aqi = df_aqi.groupby('Month')['AQI'].agg(['median', 'mean']).round(1)\n",
    "month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n",
    "\n",
    "print('Yearly Median / Mean AQI:')\n",
    "print(yearly_aqi.to_string())\n",
    "print()\n",
    "print('Monthly Median / Mean AQI:')\n",
    "print(monthly_aqi.rename(index=dict(enumerate(month_labels, 1))).to_string())\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Yearly trend\n",
    "ax = axes[0]\n",
    "ax.plot(yearly_aqi.index, yearly_aqi['median'], marker='o', color='steelblue',\n",
    "        linewidth=2, markersize=7, label='Median AQI')\n",
    "ax.fill_between(yearly_aqi.index, yearly_aqi['median'], alpha=0.15, color='steelblue')\n",
    "for yr, row in yearly_aqi.iterrows():\n",
    "    ax.annotate(f'{row[\"median\"]:.0f}', (yr, row['median']),\n",
    "                textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)\n",
    "ax.set_xlabel('Year')\n",
    "ax.set_ylabel('Median AQI')\n",
    "ax.set_title('Year-on-Year Median AQI Trend')\n",
    "ax.set_xticks(yearly_aqi.index)\n",
    "ax.annotate('* 2020 data ends July', xy=(2020, yearly_aqi.loc[2020,'median']),\n",
    "            xytext=(-60, -20), textcoords='offset points', fontsize=8, color='grey',\n",
    "            arrowprops=dict(arrowstyle='->', color='grey'))\n",
    "\n",
    "# Monthly pattern\n",
    "ax2 = axes[1]\n",
    "ax2.plot(range(1, 13), monthly_aqi['median'], marker='s', color='darkorange',\n",
    "         linewidth=2, markersize=7, label='Median AQI')\n",
    "ax2.fill_between(range(1, 13), monthly_aqi['median'], alpha=0.15, color='darkorange')\n",
    "ax2.set_xticks(range(1, 13))\n",
    "ax2.set_xticklabels(month_labels)\n",
    "ax2.set_xlabel('Month')\n",
    "ax2.set_ylabel('Median AQI')\n",
    "ax2.set_title('Monthly Median AQI Pattern (all years combined)')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(code("eda-heatmap-ym-code", [
    "# ── Year × Month AQI heatmap ─────────────────────────────────────────────────\n",
    "ym_pivot = df_aqi.groupby(['Year', 'Month'])['AQI'].median().unstack()\n",
    "ym_pivot.columns = month_labels\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(13, 4))\n",
    "sns.heatmap(ym_pivot, cmap='YlOrRd', annot=True, fmt='.0f',\n",
    "            linewidths=0.4, linecolor='white', ax=ax,\n",
    "            cbar_kws={'label': 'Median AQI'})\n",
    "ax.set_title('Year × Month Heatmap of Median AQI')\n",
    "ax.set_xlabel('Month')\n",
    "ax.set_ylabel('Year')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("eda-temporal-interp-md", [
    "**Interpretation:**  \n",
    "**Year-on-year trend:** Median AQI has declined consistently from 175 (2015) to 93 (2020). "
    "However, the 2020 data covers only January–July, which captures the post-monsoon winter peak "
    "(January) but misses the full year. The apparent improvement is therefore partly attributable "
    "to the 2020 COVID-19 lockdowns reducing industrial and vehicular emissions, and partly to "
    "incomplete year coverage.  \n",
    "\n",
    "**Monthly pattern:** A strong U-shaped pattern is visible. AQI peaks in winter "
    "(November–January, median ~183–185), drops sharply through the monsoon (July–August, ~85–86), "
    "and rises again in post-monsoon October (130). "
    "This is consistent with: (i) stagnant winter air that traps pollutants, "
    "(ii) crop-residue burning in October–November, and "
    "(iii) monsoon rains washing pollutants from the atmosphere in summer.  \n",
    "\n",
    "The year × month heatmap confirms this pattern holds across all years, with the highest cells "
    "consistently in Jan–Feb and Nov–Dec."
]))

# ── 10.6 Seasonal Analysis ────────────────────────────────────────────────────
cells.append(md("eda-seasonal-md", [
    "### 10.6 Seasonal Analysis\n",
    "\n",
    "A `Season` feature is defined using standard Indian meteorological groupings:\n",
    "\n",
    "| Season | Months | Characteristics |\n",
    "|---|---|---|\n",
    "| **Winter** | Dec, Jan, Feb | Cold, stable air; trapped pollutants; crop burning in N. India |\n",
    "| **Spring / Pre-Monsoon** | Mar, Apr, May | Warming, drier; dust storms in western India |\n",
    "| **Monsoon** | Jun, Jul, Aug, Sep | Rain washes pollutants; lowest AQI period |\n",
    "| **Post-Monsoon / Autumn** | Oct, Nov | Harvest burning; fog begins; AQI rises sharply |"
]))

cells.append(code("eda-seasonal-code", [
    "# ── Season feature + AQI comparison ─────────────────────────────────────────\n",
    "def assign_season(month):\n",
    "    if month in [12, 1, 2]:    return 'Winter'\n",
    "    if month in [3, 4, 5]:     return 'Spring/Pre-Monsoon'\n",
    "    if month in [6, 7, 8, 9]:  return 'Monsoon'\n",
    "    return 'Post-Monsoon/Autumn'\n",
    "\n",
    "df['Season']     = df['Month'].map(assign_season)\n",
    "df_aqi['Season'] = df_aqi['Month'].map(assign_season)\n",
    "\n",
    "season_order = ['Winter', 'Spring/Pre-Monsoon', 'Monsoon', 'Post-Monsoon/Autumn']\n",
    "\n",
    "season_stats = (\n",
    "    df_aqi.groupby('Season')['AQI']\n",
    "    .agg(Median='median', Mean='mean', Std='std', Count='count')\n",
    "    .round(1)\n",
    "    .reindex(season_order)\n",
    ")\n",
    "print('AQI by Season:')\n",
    "print(season_stats.to_string())\n",
    "\n",
    "# Box plot by season\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "season_colors = ['#4e79a7', '#f28e2c', '#59a14f', '#e15759']\n",
    "data_by_season = [df_aqi[df_aqi['Season'] == s]['AQI'].values for s in season_order]\n",
    "bp = ax.boxplot(data_by_season, labels=season_order, patch_artist=True,\n",
    "                medianprops=dict(color='black', linewidth=2),\n",
    "                flierprops=dict(marker='o', markersize=2, alpha=0.2, markerfacecolor='grey'))\n",
    "for patch, color in zip(bp['boxes'], season_colors):\n",
    "    patch.set_facecolor(color)\n",
    "    patch.set_alpha(0.7)\n",
    "ax.set_ylabel('AQI')\n",
    "ax.set_title('AQI Distribution by Season')\n",
    "for val, label, clr in [(100,'Satisfactory','#2ca02c'),\n",
    "                          (200,'Moderate','#ff7f0e'),\n",
    "                          (300,'Poor','#d62728')]:\n",
    "    ax.axhline(val, linestyle='--', linewidth=0.9, color=clr, alpha=0.6, label=label)\n",
    "ax.legend(fontsize=8)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("eda-seasonal-interp-md", [
    "**Interpretation:**  \n",
    "The Monsoon season has by far the lowest median AQI (89 — Satisfactory), while Winter and "
    "Post-Monsoon seasons both exceed the Moderate threshold (medians 171 and 154). "
    "The Winter season also shows the highest variance, driven by extreme pollution events in "
    "northern cities. The Monsoon–Winter contrast is approximately 2× in median AQI terms "
    "(89 vs. 171), reflecting the dominant role of rainfall and atmospheric stability in "
    "determining pollution levels."
]))

# ── 10.7 Pollutant Distributions ─────────────────────────────────────────────
cells.append(md("eda-poll-dist-md", [
    "### 10.7 Pollutant Distributions\n",
    "\n",
    "The eleven retained pollutant columns span very different concentration scales and units. "
    "Several are strongly right-skewed. Visualisations use a **log₁₀ scale on the x-axis** "
    "where distributions are heavily skewed, making the full range visible without distorting "
    "the bulk of the data. The underlying data values are **not transformed** — only the axis "
    "is scaled for display."
]))

cells.append(code("eda-poll-dist-code", [
    "# ── Pollutant distribution histograms (log x-axis for skewed cols) ───────────\n",
    "POLL11 = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene']\n",
    "# Columns with heavy right skew benefit from log x-axis\n",
    "log_cols = {'PM2.5','PM10','NO','NOx','NH3','CO','Benzene','Toluene'}\n",
    "\n",
    "fig, axes = plt.subplots(3, 4, figsize=(18, 12))\n",
    "axes = axes.flatten()\n",
    "\n",
    "for i, col in enumerate(POLL11):\n",
    "    ax = axes[i]\n",
    "    data = df[col].dropna()\n",
    "    # Shift to avoid log(0); only needed for cols with 0 values\n",
    "    if col in log_cols:\n",
    "        data_plot = data[data > 0]\n",
    "        ax.hist(data_plot, bins=50, log=False, color='steelblue', edgecolor='white', alpha=0.8)\n",
    "        ax.set_xscale('log')\n",
    "    else:\n",
    "        ax.hist(data, bins=50, color='steelblue', edgecolor='white', alpha=0.8)\n",
    "    med = data.median()\n",
    "    ax.axvline(med, color='darkred', linestyle='--', linewidth=1.3)\n",
    "    ax.set_title(f'{col}  (median={med:.1f})', fontsize=9)\n",
    "    ax.set_xlabel(f'{col} (µg/m³)' if col not in ['CO'] else 'CO (mg/m³)', fontsize=8)\n",
    "    ax.set_ylabel('Count', fontsize=8)\n",
    "    ax.tick_params(labelsize=7)\n",
    "\n",
    "# Hide unused subplot\n",
    "axes[-1].set_visible(False)\n",
    "\n",
    "plt.suptitle('Pollutant Concentration Distributions\\n'\n",
    "             '(log x-axis where right-skewed; red dashed line = median)',\n",
    "             y=1.01, fontsize=12)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("eda-poll-dist-interp-md", [
    "**Interpretation:**  \n",
    "All pollutants except O3 are right-skewed. PM2.5 and PM10 span three orders of magnitude "
    "(roughly 1–1,000 µg/m³), reflecting the enormous range from clean hill cities to heavily "
    "industrialised centres. CO and Benzene are particularly concentrated near zero with long tails. "
    "O3 has a comparatively symmetric distribution, centred around 30 µg/m³. "
    "These distributions confirm that median-based imputation (Stage 2) was appropriate — "
    "mean-based imputation would have been pulled upward by extreme values."
]))

# ── 10.8 Pollutant vs AQI Relationships ──────────────────────────────────────
cells.append(md("eda-corr-md", [
    "### 10.8 Pollutant–AQI Relationships\n",
    "\n",
    "Understanding which pollutants are most strongly associated with AQI is essential for "
    "both interpretation and feature selection in the ML stage. "
    "We first compute Pearson correlations, then examine scatter plots for the strongest predictors.  \n",
    "\n",
    "**Important:** Correlation measures linear association. "
    "A high correlation between a pollutant and AQI does not imply that the pollutant *causes* "
    "the AQI value — multiple pollutants are typically co-emitted, and AQI itself is computed "
    "from a sub-index formula rather than a simple weighted sum."
]))

cells.append(code("eda-corr-code", [
    "# ── Correlation heatmap ──────────────────────────────────────────────────────\n",
    "corr_cols = POLL11 + ['AQI']\n",
    "corr_matrix = df_aqi[corr_cols].corr().round(3)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(11, 9))\n",
    "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # upper triangle mask\n",
    "sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,\n",
    "            annot=True, fmt='.2f', linewidths=0.4, linecolor='white',\n",
    "            ax=ax, cbar_kws={'label': 'Pearson r'},\n",
    "            vmin=-0.4, vmax=1.0)\n",
    "ax.set_title('Pearson Correlation Matrix — Pollutants and AQI\\n'\n",
    "             '(lower triangle only; AQI = target variable)')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Print AQI correlations ranked\n",
    "aqi_corr = corr_matrix['AQI'].drop('AQI').sort_values(ascending=False)\n",
    "print('Pearson correlation with AQI (ranked):')\n",
    "for col, r in aqi_corr.items():\n",
    "    bar = '█' * int(abs(r) * 20)\n",
    "    print(f'  {col:<10}: {r:+.3f}  {bar}')"
]))

cells.append(code("eda-scatter-code", [
    "# ── Scatter plots: top-4 correlated pollutants vs AQI ────────────────────────\n",
    "top_poll = ['CO', 'PM2.5', 'NO2', 'PM10']\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(13, 10))\n",
    "axes = axes.flatten()\n",
    "\n",
    "for i, col in enumerate(top_poll):\n",
    "    ax = axes[i]\n",
    "    sample = df_aqi[['AQI', col]].dropna().sample(\n",
    "        n=min(3000, len(df_aqi)), random_state=42)\n",
    "    ax.scatter(sample[col], sample['AQI'],\n",
    "               alpha=0.25, s=8, color='steelblue', edgecolors='none')\n",
    "    # Trend line using numpy polyfit\n",
    "    z = np.polyfit(sample[col], sample['AQI'], 1)\n",
    "    p = np.poly1d(z)\n",
    "    xs = np.linspace(sample[col].min(), sample[col].quantile(0.98), 100)\n",
    "    ax.plot(xs, p(xs), 'r-', linewidth=1.5, label='Linear trend')\n",
    "    r = df_aqi[[col, 'AQI']].corr().loc[col, 'AQI']\n",
    "    ax.set_xlabel(col)\n",
    "    ax.set_ylabel('AQI')\n",
    "    ax.set_title(f'AQI vs {col}  (r = {r:.3f})')\n",
    "    ax.legend(fontsize=8)\n",
    "    # Cap x-axis at 99th percentile for readability\n",
    "    ax.set_xlim(0, sample[col].quantile(0.99))\n",
    "\n",
    "plt.suptitle('AQI vs Top-4 Correlated Pollutants (3,000-point sample; trend line = linear fit)',\n",
    "             y=1.02, fontsize=11)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(code("eda-scatter-extra-code", [
    "# ── Scatter: remaining notable pollutants ─────────────────────────────────────\n",
    "other_poll = ['SO2', 'NOx', 'NO', 'O3']\n",
    "\n",
    "fig, axes = plt.subplots(1, 4, figsize=(18, 4))\n",
    "\n",
    "for i, col in enumerate(other_poll):\n",
    "    ax = axes[i]\n",
    "    sample = df_aqi[['AQI', col]].dropna().sample(\n",
    "        n=min(2000, len(df_aqi)), random_state=42)\n",
    "    ax.scatter(sample[col], sample['AQI'],\n",
    "               alpha=0.2, s=7, color='darkorange', edgecolors='none')\n",
    "    z = np.polyfit(sample[col], sample['AQI'], 1)\n",
    "    xs = np.linspace(sample[col].min(), sample[col].quantile(0.98), 100)\n",
    "    ax.plot(xs, np.poly1d(z)(xs), 'b-', linewidth=1.4)\n",
    "    r = df_aqi[[col, 'AQI']].corr().loc[col, 'AQI']\n",
    "    ax.set_xlabel(col)\n",
    "    ax.set_ylabel('AQI')\n",
    "    ax.set_title(f'{col}  (r = {r:.3f})')\n",
    "    ax.set_xlim(0, sample[col].quantile(0.99))\n",
    "\n",
    "plt.suptitle('AQI vs SO2, NOx, NO, O3 (sample; blue = linear trend)',\n",
    "             y=1.02, fontsize=11)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("eda-corr-interp-md", [
    "**Interpretation:**  \n",
    "**CO** (r = 0.678) and **PM2.5** (r = 0.657) are the two pollutants most strongly correlated "
    "with AQI. **NO2** (r = 0.535), **PM10** (r = 0.494), **SO2** (r = 0.486), and **NOx** "
    "(r = 0.468) also show moderate positive correlations.  \n",
    "\n",
    "**Benzene** (r = 0.049) and **NH3** (r = 0.092) have near-zero correlations with AQI, "
    "suggesting they are less informative as standalone predictors.  \n",
    "\n",
    "The scatter plots show that while a positive trend is clear, the relationships are non-linear "
    "and highly scattered — especially for extreme AQI values. "
    "This suggests that a non-linear model (e.g., Random Forest) should outperform "
    "simple linear regression.  \n",
    "\n",
    "Note also that NOx is highly correlated with NO (r ≈ 0.79), and Benzene is highly correlated "
    "with Toluene (r ≈ 0.71), indicating multicollinearity that will need attention in the ML stage."
]))

# ── 10.9 City × Pollutant Analysis ───────────────────────────────────────────
cells.append(md("eda-city-poll-md", [
    "### 10.9 City × Pollutant Analysis\n",
    "\n",
    "To understand what drives city-level AQI differences, we examine how median concentrations "
    "of key pollutants vary across cities. Each cell in the heatmap shows the median value of "
    "a pollutant for a given city, scaled column-wise (z-score) so that cities can be compared "
    "relative to the national median regardless of the absolute units of each pollutant."
]))

cells.append(code("eda-city-poll-code", [
    "# ── City × Pollutant heatmap (z-score normalised per column) ─────────────────\n",
    "focus_polls = ['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'SO2', 'O3', 'NH3']\n",
    "\n",
    "city_poll_med = (\n",
    "    df.groupby('City')[focus_polls]\n",
    "    .median()\n",
    "    .round(1)\n",
    ")\n",
    "# Z-score normalise each column\n",
    "city_poll_z = (city_poll_med - city_poll_med.mean()) / city_poll_med.std()\n",
    "\n",
    "# Sort cities by their mean normalised score (overall pollution level)\n",
    "city_poll_z = city_poll_z.loc[city_stats.index]  # match AQI-sorted order\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 9))\n",
    "sns.heatmap(city_poll_z, cmap='RdYlGn_r', center=0,\n",
    "            annot=city_poll_med.loc[city_poll_z.index, focus_polls],\n",
    "            fmt='.0f', linewidths=0.3, linecolor='white',\n",
    "            ax=ax, cbar_kws={'label': 'Z-score (column-normalised)'},\n",
    "            annot_kws={'size': 7})\n",
    "ax.set_title('City × Pollutant Heatmap\\n'\n",
    "             'Colour = z-score; cell values = median concentration (µg/m³ or mg/m³ for CO)\\n'\n",
    "             'Cities sorted top-to-bottom by median AQI (cleanest → most polluted)')\n",
    "ax.set_xlabel('Pollutant')\n",
    "ax.set_ylabel('City')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print('\\nCity × Pollutant Median Concentrations (raw values):')\n",
    "print(city_poll_med.loc[city_poll_z.index].to_string())"
]))

cells.append(md("eda-city-poll-interp-md", [
    "**Interpretation:**  \n",
    "The heatmap reveals distinct pollution profiles across cities:  \n",
    "\n",
    "- **Delhi and Gurugram** stand out with extremely high PM2.5 and PM10 — Delhi's median PM2.5 "
    "is 94.6 µg/m³, nearly 4× the national median.  \n",
    "- **Ahmedabad** has uniquely high CO (16.2 mg/m³) and SO2 (46.8 µg/m³), suggesting a "
    "different emission source mix (industrial/combustion-heavy) compared to Delhi's "
    "particulate-dominated profile.  \n",
    "- **Kochi** shows anomalously high NO (68.6 µg/m³) despite a moderate overall AQI — "
    "possibly reflecting local traffic or industrial NO sources that do not translate "
    "proportionally to AQI.  \n",
    "- **Aizawl and Shillong** (north-east India, hilly terrain) have consistently low "
    "concentrations across all pollutants.  \n",
    "- **SO2 is elevated in industrial cities** (Ahmedabad, Talcher, Jorapokhar) — consistent "
    "with coal-fired power plants and heavy industries in those locations."
]))

# ── 10.10 Key EDA Findings ────────────────────────────────────────────────────
cells.append(md("eda-findings-md", [
    "### 10.10 Key Findings from EDA\n",
    "\n",
    "The following findings are all grounded in quantitative results from the analysis above.\n",
    "\n",
    "---\n",
    "\n",
    "**1. Strongly right-skewed AQI distribution**  \n",
    "AQI has a median of 118 (Moderate) but a mean of 167, pulled upward by extreme events "
    "(max = 2,049). Skewness = 3.4. This means the majority of days are in the Moderate or "
    "Satisfactory range, but severe episodes can be far beyond the nominal scale maximum of 500.\n",
    "\n",
    "**2. Most days are Moderate or Satisfactory**  \n",
    "68.6% of observations fall in the Moderate (35.5%) or Satisfactory (33.1%) categories. "
    "Only 5.4% are classified as Good — as rare as the Severe category (5.4%).\n",
    "\n",
    "**3. Extreme city-level disparity**  \n",
    "Median AQI ranges from 23 (Aizawl — Good) to 385 (Ahmedabad — Very Poor). "
    "Delhi (median 257), Patna (215), Gurugram (209), and Lucknow (198) form a high-pollution "
    "cluster, while southern and north-eastern cities are substantially cleaner.\n",
    "\n",
    "**4. Consistent declining trend (2015–2019), disrupted in 2020**  \n",
    "Median AQI fell from 175 (2015) to 109 (2019), a 38% reduction over four years. "
    "The 2020 median of 93 must be interpreted cautiously — data ends in July, "
    "missing the autumn/winter peaks, and COVID lockdowns suppressed emissions.\n",
    "\n",
    "**5. Strong seasonal cycle**  \n",
    "Monsoon AQI (median 89) is roughly half of Winter AQI (median 171). "
    "Post-monsoon October–November shows a sharp AQI spike associated with crop-residue burning "
    "and the onset of winter atmospheric conditions.\n",
    "\n",
    "**6. CO and PM2.5 are the strongest AQI predictors**  \n",
    "Pearson correlation with AQI: CO (r = 0.678), PM2.5 (r = 0.657), NO2 (r = 0.535), "
    "PM10 (r = 0.494). Benzene (r = 0.049) and NH3 (r = 0.092) show near-zero linear association.\n",
    "\n",
    "**7. Non-linear pollutant–AQI relationships**  \n",
    "Scatter plots show that while positive trends exist, the relationship between pollutants "
    "and AQI is non-linear and highly variable — particularly at extreme values. "
    "This motivates the use of non-linear ML models (e.g., Random Forest) in Stage 5.\n",
    "\n",
    "**8. Multicollinearity between pollutant pairs**  \n",
    "NOx and NO are highly correlated (r ≈ 0.79); Benzene and Toluene (r ≈ 0.71). "
    "In the ML stage, feature selection or regularisation will be needed to address this.\n",
    "\n",
    "**9. Distinct city pollution profiles**  \n",
    "Delhi's pollution is particulate-dominated (high PM2.5, PM10); Ahmedabad's is "
    "combustion-dominated (very high CO, SO2); Kochi shows anomalously high NO. "
    "These differences suggest that a single national model may benefit from city as a feature.\n",
    "\n",
    "**10. Data coverage is uneven across cities**  \n",
    "Observation counts range from 111 (Aizawl) to 1,999 (Delhi). "
    "City-level conclusions for sparsely covered cities should be interpreted with caution."
]))

# ── Inject cells ──────────────────────────────────────────────────────────────
nb['cells'].extend(cells)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Injected {len(cells)} new EDA cells.')
print(f'New total cell count: {len(nb["cells"])}')
code_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
md_count   = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
print(f'Code cells: {code_count}  |  Markdown cells: {md_count}')
