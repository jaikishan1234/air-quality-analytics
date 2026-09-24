import json
with open('notebooks/JaikishanNayak_AirQualityAnalytics.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
# Find cells with dev-artifact IDs
print('Cells with dev-artifact-like IDs:')
for i, c in enumerate(nb['cells']):
    cid = c.get('id','')
    if any(x in cid for x in ['verif-','_raw','probe','inject','verify']):
        src = ''.join(c['source'])[:70].replace('\n',' ')
        ct  = c['cell_type']
        print(f'  [{i+1:3d}] id={cid}  type={ct}  {src}')

# Also check for any cell that references internal dev artifacts in its source
print()
print('Cells referencing dev artifacts in source:')
keywords = ['_nb_inspect', '_inject', '_probe', '_verify', 'scripts/', 'TODO', 'FIXME']
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    for kw in keywords:
        if kw in src:
            print(f'  [{i+1:3d}] [{c["cell_type"]}] keyword={kw}  {src[:60].replace(chr(10)," ")}')
            break
