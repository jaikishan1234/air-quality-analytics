import json

with open('notebooks/JaikishanNayak_AirQualityAnalytics.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
ml_cells   = [c for c in code_cells if str(c.get('id','')).startswith('ml-')]

print(f'Total cells        : {len(nb["cells"])}')
print(f'Code cells total   : {len(code_cells)}')
print(f'Stage-5 code cells : {len(ml_cells)}')
print()

all_exec = all(c.get('execution_count') is not None for c in ml_cells)
print(f'All Stage-5 cells executed: {all_exec}')
print()

img_count = sum(1 for c in ml_cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{}))
print(f'Stage-5 image outputs: {img_count}')
print()

for c in ml_cells:
    n_out = len(c.get('outputs', []))
    ec    = c.get('execution_count')
    has_img = any('image/png' in o.get('data',{}) for o in c.get('outputs',[]))
    flag = '[IMG]' if has_img else '     '
    src  = ''.join(c['source'])[:60].replace('\n',' ')
    print(f'  {flag} ec={str(ec):3s} out={n_out}  {src}')
    for o in c.get('outputs',[]):
        if o.get('output_type') == 'stream':
            txt = ''.join(o.get('text',[]))
            for line in txt.strip().split('\n')[:12]:
                print(f'         | {line}')
    print()
