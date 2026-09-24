import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'ml-actual-pred-code':
        # Completely replace source with a clean, safe version
        new_source_lines = [
            "# ── Actual vs Predicted scatter plot ─────────────────────────────────────────\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n",
            "\n",
            "# --- Panel 1: Actual vs Predicted ---\n",
            "ax = axes[0]\n",
            "ax.scatter(y_test, y_pred_best, alpha=0.2, s=6, color='steelblue', edgecolors='none')\n",
            "\n",
            "# 45-degree perfect-prediction reference line\n",
            "lim = max(y_test.max(), y_pred_best.max()) * 1.05\n",
            "ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Perfect prediction (y=x)')\n",
            "ax.set_xlim(0, lim)\n",
            "ax.set_ylim(0, lim)\n",
            "ax.set_xlabel('Actual AQI')\n",
            "ax.set_ylabel('Predicted AQI')\n",
            "ax.set_title('Actual vs Predicted AQI   ' + best_model_name)\n",
            "ax.legend(fontsize=8)\n",
            "\n",
            "# Annotation box with key metrics\n",
            "_ann = 'R2=' + str(round(float(best_row['R2']),4)) + '  RMSE=' + str(round(float(best_row['RMSE']),2)) + '  MAE=' + str(round(float(best_row['MAE']),2))\n",
            "ax.text(0.05, 0.94, _ann, transform=ax.transAxes, fontsize=9,\n",
            "        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n",
            "\n",
            "# --- Panel 2: Residual distribution ---\n",
            "ax2 = axes[1]\n",
            "ax2.hist(residuals, bins=60, color='darkorange', edgecolor='white', alpha=0.8, density=True)\n",
            "pd.Series(residuals).plot.kde(ax=ax2, color='darkred', linewidth=2)\n",
            "ax2.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Zero residual')\n",
            "_mean_resid = round(float(residuals.mean()), 1)\n",
            "ax2.axvline(_mean_resid, color='steelblue', linestyle='--', linewidth=1.2,\n",
            "            label='Mean residual = ' + str(_mean_resid))\n",
            "ax2.set_xlabel('Residual (Actual - Predicted AQI)')\n",
            "ax2.set_ylabel('Density')\n",
            "ax2.set_title('Residual Distribution')\n",
            "ax2.legend(fontsize=8)\n",
            "\n",
            "plt.suptitle('Prediction Diagnostics -- ' + best_model_name, y=1.02, fontsize=12)\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ]
        cell['source'] = new_source_lines
        print('Replaced ml-actual-pred-code with clean version.')
        break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
