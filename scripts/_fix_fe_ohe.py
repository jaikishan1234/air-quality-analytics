import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'fe-selection-code':
        src = ''.join(cell['source'])
        # Replace the problematic dynamic reference with the literal known value
        src = src.replace(
            'Total after OHE     : {len(NUMERIC_FEATURES) + len(ohe_cols)}',
            'Total after OHE     : 42  (20 numeric + 22 city OHE)'
        )
        cell['source'] = [line + '\n' for line in src.split('\n')]
        cell['source'][-1] = cell['source'][-1].rstrip('\n')
        print('Fixed fe-selection-code')
        break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
