import json

with open('notebooks/JaikishanNayak_AirQualityAnalytics.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
eda_cells  = [c for c in code_cells if str(c.get('id','')).startswith('eda-')]

print(f'Total cells       : {len(nb["cells"])}')
print(f'Code cells total  : {len(code_cells)}')
print(f'EDA code cells    : {len(eda_cells)}')
print()

# Count images
img_total = sum(
    1 for c in code_cells
    for o in c.get('outputs', [])
    if 'image/png' in o.get('data', {})
)
print(f'Total image outputs in notebook: {img_total}')
print()

executed = [c for c in code_cells if c.get('execution_count') is not None]
print(f'Executed code cells: {len(executed)}')
print()

print('EDA cell execution summary:')
for c in eda_cells:
    n_out = len(c.get('outputs', []))
    ec    = c.get('execution_count')
    has_img = any('image/png' in o.get('data',{}) for o in c.get('outputs',[]))
    src = ''.join(c['source'])[:55].replace('\n',' ')
    flag = '[IMG]' if has_img else '     '
    print(f'  {flag} [{c["id"]}] ec={ec} out={n_out}  {src}')
