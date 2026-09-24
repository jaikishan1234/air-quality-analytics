import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'ml-actual-pred-code':
        print('Current source:')
        for i, line in enumerate(cell['source']):
            print(f'  {i:2d}: {repr(line)}')
        break
