import json
with open('notebooks/JaikishanNayak_AirQualityAnalytics.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells   = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
md_text    = ' '.join(''.join(c['source']) for c in md_cells)
executed   = sum(1 for c in code_cells if c.get('execution_count') is not None)
imgs       = sum(1 for c in code_cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{}))

checks = [
    ('Section 13 present',             '13. Model Interpretation' in md_text),
    ('13.1 Selected model',            '13.1 Selected Model' in md_text),
    ('13.2 Permutation importance',    'Permutation Importance' in md_text),
    ('13.3 Group importance',          'Feature Group Importance' in md_text),
    ('13.4 Sample pred table',         'Actual vs Predicted' in md_text and 'Sample Table' in md_text),
    ('13.5 AQI range errors',          'AQI Range Error Analysis' in md_text),
    ('13.6 City errors',               'City-Level Error Analysis' in md_text),
    ('13.7 Temporal errors',           'Temporal Error Analysis' in md_text),
    ('13.8 Worst cases',               'Worst Prediction Cases' in md_text),
    ('13.9 Limitations',               'Model Limitations' in md_text),
    ('13.10 Final interpretation',     'Final ML Interpretation' in md_text),
    ('All 77 cells executed',          executed == 77),
    ('24 chart images',                imgs == 24),
]
all_ok = True
for lbl, ok in checks:
    status = 'PASS' if ok else 'FAIL'
    if not ok: all_ok = False
    print(status + '  ' + lbl)
print()
print('Overall: ' + ('ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'))
print('Total cells: ' + str(len(nb['cells'])) + '  Code: ' + str(len(code_cells)) + '  Markdown: ' + str(len(md_cells)))
print('Executed: ' + str(executed) + ' / ' + str(len(code_cells)) + '  Images: ' + str(imgs))
