import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'ml-resid-scatter-code':
        # Replace the set_title that had an embedded newline f-string
        new_source_lines = [
            "# ── Residual vs Predicted scatter ─────────────────────────────────────────────\n",
            "fig, ax = plt.subplots(figsize=(10, 5))\n",
            "ax.scatter(y_pred_best, residuals, alpha=0.2, s=6, color='seagreen', edgecolors='none')\n",
            "ax.axhline(0, color='red', linestyle='--', linewidth=1.5)\n",
            "ax.set_xlabel('Predicted AQI')\n",
            "ax.set_ylabel('Residual (Actual - Predicted)')\n",
            "ax.set_title('Residuals vs Predicted AQI -- ' + best_model_name)\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print('Residual statistics:')\n",
            "print('  Mean   :', round(float(residuals.mean()), 2))\n",
            "print('  Std    :', round(float(residuals.std()), 2))\n",
            "print('  Min    :', round(float(residuals.min()), 2))\n",
            "print('  Max    :', round(float(residuals.max()), 2))\n",
            "print('  Skew   :', round(float(pd.Series(residuals).skew()), 3))",
        ]
        cell['source'] = new_source_lines
        print('Replaced ml-resid-scatter-code')
        break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
