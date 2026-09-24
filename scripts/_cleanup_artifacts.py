import json

NB_PATH = 'notebooks/JaikishanNayak_AirQualityAnalytics.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── 1. Remove Stage 2 Verification cells (cells with id starting with 'verif-') ──
before = len(nb['cells'])
nb['cells'] = [c for c in nb['cells'] if not c.get('id','').startswith('verif-')]
after = len(nb['cells'])
print(f'Removed {before - after} Stage-2 verification cells. Cells: {before} -> {after}')

# ── 2. Remove the "12.10 Reusable Variables for Stage 6" markdown (staging note) ──
# and the "11.10 Preparation for Model Training" block (internal staging)
# and the "13.11 Stage Output" block (internal staging)
# Keep the confirmation code cells that actually print useful results - only remove
# the pure-staging internal markdown sections that say "leave variables for next stage"
staging_ids_to_remove = {
    'ml-ready-md',      # "11.10 / 12.10 Preparation for Model Training"
    'interp-vars-md',   # "13.11 Stage Output — Reusable Variables"
}
before = len(nb['cells'])
nb['cells'] = [c for c in nb['cells'] if c.get('id','') not in staging_ids_to_remove]
after = len(nb['cells'])
print(f'Removed {before - after} staging-note markdown cells. Cells: {before} -> {after}')

# ── 3. Verify the remaining structure ──
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells   = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
print(f'Remaining: {len(nb["cells"])} total  ({len(code_cells)} code, {len(md_cells)} md)')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
