
import pandas as pd
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path("/Users/dede/Lab/adom-tear-film-theoretical-spectra-generator")

file_path = PROJECT_ROOT / "exploration" / "more_good_spectras" / "Lipid  and  Mucus-Aqueous_Height.xlsx"
df = pd.read_excel(file_path)
print(df.head(10))
print(df.columns)
