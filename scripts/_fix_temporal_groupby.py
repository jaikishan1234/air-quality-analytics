import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'interp-temporal-code':
        cell['source'] = [
            "# ── Monthly error on test set ─────────────────────────────────────────────────\n",
            "pred_table['Month_num'] = df_test['Date'].dt.month.values\n",
            "\n",
            "month_labels_map = {12:'Dec-19', 1:'Jan-20', 2:'Feb-20', 3:'Mar-20',\n",
            "                    4:'Apr-20', 5:'May-20', 6:'Jun-20', 7:'Jul-20'}\n",
            "\n",
            "# groupby key (Month_num) is excluded from group when include_groups=False;\n",
            "# use g.name to access the group key inside the lambda\n",
            "temporal_error = (\n",
            "    pred_table.groupby('Month_num')\n",
            "    .apply(lambda g: pd.Series({\n",
            "        'Month':       month_labels_map.get(g.name, str(g.name)),\n",
            "        'N':           len(g),\n",
            "        'MAE':         round(g['Abs_Error'].mean(), 2),\n",
            "        'RMSE':        round(np.sqrt((g['Residual']**2).mean()), 2),\n",
            "        'Mean_Actual': round(g['Actual_AQI'].mean(), 1),\n",
            "    }), include_groups=False)\n",
            "    .reset_index()\n",
            "    .sort_values('Month_num')\n",
            ")\n",
            "\n",
            "print('Monthly error on test set:')\n",
            "print(temporal_error[['Month','N','MAE','RMSE','Mean_Actual']].to_string(index=False))\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 4))\n",
            "ax.plot(temporal_error['Month'], temporal_error['MAE'],  marker='o', label='MAE',\n",
            "        color='steelblue', linewidth=2, markersize=7)\n",
            "ax.plot(temporal_error['Month'], temporal_error['RMSE'], marker='s', label='RMSE',\n",
            "        color='darkorange', linewidth=2, markersize=7)\n",
            "for _, row in temporal_error.iterrows():\n",
            "    ax.annotate(str(row['MAE']), (row['Month'], row['MAE']),\n",
            "                textcoords='offset points', xytext=(0, 6), ha='center', fontsize=7.5, color='steelblue')\n",
            "ax.set_xlabel('Month (test set: Dec 2019 to Jul 2020)')\n",
            "ax.set_ylabel('Error (AQI units)')\n",
            "ax.set_title('Monthly MAE and RMSE on Test Set (Jul-20 has only 24 obs)')\n",
            "ax.legend()\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ]
        print('Fixed interp-temporal-code')
        break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
