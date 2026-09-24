import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix the fe-selection-md cell to reflect actual 22 OHE columns (23 training cities)
for cell in nb['cells']:
    if cell.get('id') == 'fe-selection-md':
        src = ''.join(cell['source'])
        src = src.replace(
            '| **City** | One-hot encoded (26 cities, drop_first → 25 binary columns) | 25 |',
            '| **City** | One-hot encoded (23 cities present in training, drop_first → 22 binary columns) | 22 |'
        )
        src = src.replace(
            '**Total numeric + temporal input features before OHE:** 20  \n**Total features after OHE (with drop_first):** 45',
            '**Total numeric + temporal input features before OHE:** 20  \n**Total features after OHE (with drop_first):** 42  \n*(3 cities — Aizawl, Ernakulam, Kochi — appear only in the test period; they are absent from the training OHE fit but handled via `handle_unknown=\'ignore\'`)*'
        )
        cell['source'] = [line + '\n' for line in src.split('\n')]
        cell['source'][-1] = cell['source'][-1].rstrip('\n')
        print('Fixed fe-selection-md')
        break

# Also fix fe-selection-code total count line
for cell in nb['cells']:
    if cell.get('id') == 'fe-selection-code':
        new_src = []
        for line in cell['source']:
            if "City (OHE, drop=1st): 25  (from 26 unique cities)" in line:
                line = line.replace(
                    "City (OHE, drop=1st): 25  (from 26 unique cities)",
                    "City (OHE, drop=1st): 22  (23 training cities, drop_first)"
                )
            if "Total after OHE     : {len(NUMERIC_FEATURES) + 25}" in line:
                line = line.replace(
                    "Total after OHE     : {len(NUMERIC_FEATURES) + 25}",
                    "Total after OHE     : {len(NUMERIC_FEATURES) + len(ohe_cols)}"
                )
            new_src.append(line)
        cell['source'] = new_src
        print('Fixed fe-selection-code')
        break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Saved.')
