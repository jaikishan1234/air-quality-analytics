import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'ml-actual-pred-code':
        src = ''.join(cell['source'])
        # Fix the multi-line f-string title that breaks the kernel
        src = src.replace(
            "ax.set_title(f'Actual vs Predicted AQI\n{best_model_name}  (R²={best_row[\"R2\"]:.4f})')",
            "ax.set_title('Actual vs Predicted AQI\\n' + best_model_name + '  (R\u00b2=' + str(round(best_row[\"R2\"],4)) + ')')"
        )
        cell['source'] = [l + '\n' for l in src.split('\n')]
        cell['source'][-1] = cell['source'][-1].rstrip('\n')
        print('Fixed ml-actual-pred-code')
        break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
