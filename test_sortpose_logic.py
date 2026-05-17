import pandas as pd
import numpy as np

def compute_iqr_margins(df, iqr_scale_factor):
    margins = {}
    for col in ['pitch', 'yaw', 'roll']:
        q75, q25 = np.percentile(df[col], [75, 25])
        iqr = q75 - q25
        margins[col] = iqr * iqr_scale_factor
    return margins

# 1. Create a tiny pandas DataFrame
data = {
    'pitch': [10, 20, 30, 40], # IQR = 30-20 = 10 (using midpoints for percentiles can vary, but numpy default is linear)
    'yaw': [0, 5, 10, 15],     # IQR = 11.25 - 3.75 = 7.5
    'roll': [5, 5, 5, 5]       # IQR = 0
}
df = pd.DataFrame(data)

# 2. IQR Scale Factor
iqr_scale_factor = 1.5

# 3. Compute margins
margins = compute_iqr_margins(df, iqr_scale_factor)

# Calculate expected IQRs for report
iqrs = {}
for col in ['pitch', 'yaw', 'roll']:
    q75, q25 = np.percentile(df[col], [75, 25])
    iqrs[col] = q75 - q25

print(f"Scale Factor: {iqr_scale_factor}")
print(f"Sample IQRs: {iqrs}")
print(f"Resulting Margins: {margins}")
