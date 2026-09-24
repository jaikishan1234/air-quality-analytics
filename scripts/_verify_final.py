import json

with open('notebooks/JaikishanNayak_AirQualityAnalytics.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells   = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
executed   = sum(1 for c in code_cells if c.get('execution_count') is not None)
imgs       = sum(1 for c in code_cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{}))
md_text    = ' '.join(''.join(c['source']) for c in md_cells)

print('=== FINAL NOTEBOOK STATE ===')
print('Total cells    :', len(nb['cells']))
print('Code cells     :', len(code_cells))
print('Markdown cells :', len(md_cells))
print('Executed cells :', executed, '/', len(code_cells))
print('Chart images   :', imgs)
print()

# Check for residual verification cells
verif_cells = [c for c in nb['cells'] if c.get('id','').startswith('verif-')]
print('Verification artifact cells remaining:', len(verif_cells))
print()

# Check final summary code cell output
concl_code = [c for c in code_cells if c.get('id','') == 'concl-summary-code']
if concl_code:
    c = concl_code[0]
    print('Summary code cell ec=' + str(c.get('execution_count')) + ':')
    for o in c.get('outputs',[]):
        if o.get('output_type') == 'stream':
            print(''.join(o.get('text',[])))
else:
    print('WARNING: concl-summary-code not found')

print()
# All major sections present
sections_required = [
    '1. Project Overview',
    '2. Problem Statement',
    '3. Objectives',
    '4. Dataset Description',
    '5. Import Libraries',
    '6. Load Dataset',
    '7. Data Understanding',
    '8. Data Quality Assessment',
    '9. Data Cleaning and Preprocessing',
    '10. Exploratory Data Analysis',
    '11. Feature Engineering',
    '12. AQI Prediction Models',
    '13. Model Interpretation',
    '14. Final Conclusion',
    'Project Deliverables',
]
print('Section presence check:')
all_ok = True
for s in sections_required:
    ok = s in md_text
    print('  ' + ('PASS' if ok else 'FAIL') + '  ' + s)
    if not ok: all_ok = False

print()
print('All sections present:', all_ok)
print()
print('=== SECTION HEADINGS (H2) ===')
for c in md_cells:
    src = ''.join(c['source'])
    for line in src.split('\n'):
        if line.startswith('## '):
            print(' ', line)
