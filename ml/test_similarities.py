import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('data/raw/categories_products.csv', names=['Category_Code', 'SKU_ID', 'Product_Name'], skiprows=1)
m = np.load('models/gnn/similarity_matrix.npy')

# Find frying pan
idx = df[df['SKU_ID'] == 'SKU_KICH001'].index[0]
sims = m[idx]

# Get top 15
top_idx = np.argsort(sims)[::-1][:15]

print('\n🍳 Top 15 similar products to Frying Pan (SKU_KICH001):')
print('='*70)
for i, idx2 in enumerate(top_idx, 1):
    if sims[idx2] > 0:
        sku = df.iloc[idx2]['SKU_ID']
        name = df.iloc[idx2]['Product_Name']
        cat = df.iloc[idx2]['Category_Code']
        sim = sims[idx2]
        cross = '🔗' if cat != 'KICH' else '  '
        print(f'{i:2d}. {cross} {sku:15s} {name:30s} [{cat}] → {sim:.3f}')
