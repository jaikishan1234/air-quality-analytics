import json, sys

with open('notebooks/JaikishanNayak_AirQualityAnalytics.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('nbformat:', nb.get('nbformat'), nb.get('nbformat_minor'))
print('Total cells:', len(nb['cells']))
code_cells  = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells    = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
print('Code cells:', len(code_cells))
print('Markdown cells:', len(md_cells))
print()
print('All cells (type + first 80 chars of source):')
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])[:80].replace('\n', ' ')
    n_out = len(c.get('outputs', []))
    ec    = c.get('execution_count', '-')
    print(f'  [{i+1:2d}] {c["cell_type"]:8s} ec={str(ec):3s} out={n_out}  {src}')
