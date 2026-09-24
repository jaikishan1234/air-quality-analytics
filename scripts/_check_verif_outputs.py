import json

with open('notebooks/JaikishanNayak_AirQualityAnalytics.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f'Total cells : {len(nb["cells"])}')
print(f'Code cells  : {len(code_cells)}')
print()

# Find verification cells by id prefix
verif_code = [c for c in code_cells if str(c.get('id','')).startswith('verif-')]
print(f'Verification code cells: {len(verif_code)}')
print()

for c in verif_code:
    n_out = len(c.get('outputs', []))
    ec    = c.get('execution_count')
    src   = ''.join(c['source'])[:55].replace('\n',' ')
    print(f'  [{c["id"]}] ec={ec} out={n_out}  {src}')
    for o in c.get('outputs', []):
        if o.get('output_type') == 'stream':
            text = ''.join(o.get('text', []))
            print('    >> ' + text[:400].replace('\n', '\n    >> '))
    print()
