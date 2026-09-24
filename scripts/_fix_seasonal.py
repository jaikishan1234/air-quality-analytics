import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'eda-seasonal-code':
        src = cell['source']
        # Find and fix trailing quote on last plt.show() line
        fixed = []
        for line in src:
            # Remove a trailing '"' that was accidentally appended
            if line.strip() == 'plt.show()"':
                line = line.replace('plt.show()"', 'plt.show()')
            fixed.append(line)
        cell['source'] = fixed
        print('Fixed seasonal cell. Last 3 lines:')
        for l in fixed[-3:]:
            print(repr(l))
        break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Saved.')
