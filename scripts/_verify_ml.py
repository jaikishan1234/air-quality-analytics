import json
with open('notebooks/JaikishanNayak_AirQualityAnalytics.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
code_cells = [c for c in nb['cells'] if c['cell_type']=='code']
executed   = sum(1 for c in code_cells if c.get('execution_count') is not None)
imgs       = sum(1 for c in code_cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{}))
md_text    = ' '.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='markdown')
checks = [
    ('Section 12 present',             '12. AQI Prediction Models' in md_text),
    ('12.1 Model selection',           'Model Selection' in md_text),
    ('12.3 Evaluate models',           'Evaluate All Models' in md_text),
    ('12.4 Comparison viz',            'Model Comparison Visualisation' in md_text),
    ('12.5 Baseline comparison',       'Improvement Over Baseline' in md_text),
    ('12.7 Pred vs actual',            'Prediction vs Actual' in md_text),
    ('12.8 Error analysis',            'Error Analysis' in md_text),
    ('12.9 Training summary',          'Model Training Summary' in md_text),
    ('12.10 Reusable vars',            'Reusable Variables for Stage 6' in md_text),
    ('All 67 cells executed',          executed == 67),
    ('Total images >= 18',             imgs >= 18),
]
all_ok = True
for lbl, ok in checks:
    status = 'PASS' if ok else 'FAIL'
    print(status + '  ' + lbl)
    if not ok:
        all_ok = False
print()
print('Overall: ' + ('ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'))
print('Total images in notebook: ' + str(imgs))
print('Total cells: ' + str(len(nb['cells'])))
print('Code cells executed: ' + str(executed) + ' / ' + str(len(code_cells)))
