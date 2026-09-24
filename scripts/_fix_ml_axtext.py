import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'ml-actual-pred-code':
        src = ''.join(cell['source'])

        # Fix ax.text with multi-line f-string
        old = """ax.text(0.05, 0.92, f'R² = {best_row["R2"]:.4f}\\nRMSE = {best_row["RMSE"]:.2f}\\nMAE  = {best_row["MAE"]:.2f}',"""
        new = """ax.text(0.05, 0.92, 'R2 = ' + str(round(best_row["R2"],4)) + '  RMSE = ' + str(round(best_row["RMSE"],2)) + '  MAE = ' + str(round(best_row["MAE"],2)),"""
        src = src.replace(old, new)

        cell['source'] = [l + '\n' for l in src.split('\n')]
        cell['source'][-1] = cell['source'][-1].rstrip('\n')
        print('Fixed ml-actual-pred-code ax.text')
        print('Sample:', repr(''.join(cell['source'])[:200]))
        break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
