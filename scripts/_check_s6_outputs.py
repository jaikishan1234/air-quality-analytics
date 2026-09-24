import json

with open('notebooks/JaikishanNayak_AirQualityAnalytics.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
s6_cells   = [c for c in code_cells if str(c.get('id','')).startswith('interp-')]

total_cells  = len(nb['cells'])
total_code   = len(code_cells)
total_exec   = sum(1 for c in code_cells if c.get('execution_count') is not None)
total_imgs   = sum(1 for c in code_cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{}))

print('Total cells       :', total_cells)
print('Code cells        :', total_code)
print('Executed          :', total_exec, '/', total_code)
print('Chart images      :', total_imgs)
print('Stage-6 code cells:', len(s6_cells))
print()

all_exec6 = all(c.get('execution_count') is not None for c in s6_cells)
print('All Stage-6 cells executed:', all_exec6)
print()

for c in s6_cells:
    ec    = c.get('execution_count')
    n_out = len(c.get('outputs', []))
    has_img = any('image/png' in o.get('data',{}) for o in c.get('outputs',[]))
    flag = '[IMG]' if has_img else '     '
    src  = ''.join(c['source'])[:55].replace('\n',' ')
    print(flag, 'ec=' + str(ec).ljust(3), 'out=' + str(n_out), ' ', src)
    for o in c.get('outputs',[]):
        if o.get('output_type') == 'stream':
            txt = ''.join(o.get('text',[]))
            for line in txt.strip().split('\n')[:6]:
                print('       |', line)
    print()
