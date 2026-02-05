# Shlomo ground truth (lipid / aqueous)

- **Source:** `Lipid  and  Mucus-Aqueous_Height.xlsx` (Sheet1), columns: Absolute time, Lipid_Height, Mucus-Aqueous_Height.
- **`shlomo_ground_truth.csv`:** One row per spectrum with a BestFit in this folder. Columns: `spectrum_name`, `spectrum_base`, `absolute_time`, `lipid_shlomo_nm`, `aqueous_shlomo_nm`.
- **Usage:** PyElli app and evaluation scripts use this to:
  - Show Shlomo’s lipid/aqueous next to our fit and report **lipid error** and **aqueous error** (|ours − Shlomo|).
  - Compute **loss at Shlomo’s values** (theoretical spectrum at his L/A, then score vs measured) to compare with loss at our best fit (same solution basin vs suboptimal labels / missing constraint).

To regenerate the CSV from the xlsx (e.g. after Shlomo updates the spreadsheet), run from repo root:

```bash
conda run -n adom-streamlit python -c "
import pandas as pd
from pathlib import Path
path = Path('exploration/more_good_spectras/Lipid  and  Mucus-Aqueous_Height.xlsx')
df = pd.read_excel(path, sheet_name='Sheet1')
df.columns = ['absolute_time', 'lipid_shlomo_nm', 'aqueous_shlomo_nm']
df['absolute_time'] = df['absolute_time'].astype(str).str.strip()
df['spectrum_name'] = '(Run)spectra_' + df['absolute_time'] + '.txt'
df['spectrum_base'] = '(Run)spectra_' + df['absolute_time']
out = df[['spectrum_name', 'spectrum_base', 'absolute_time', 'lipid_shlomo_nm', 'aqueous_shlomo_nm']]
out.to_csv(Path('exploration/more_good_spectras/shlomo_ground_truth.csv'), index=False)
print('Wrote shlomo_ground_truth.csv, rows:', len(out))
"
```
