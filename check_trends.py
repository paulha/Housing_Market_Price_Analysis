
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv('data/gresham/gresham_research_db.csv')
df['SaleDate'] = pd.to_datetime(df['SaleDate'])
df = df[df['RMV'] > 0].copy()
df['Ratio'] = df['SalePrice'] / df['RMV']

# 1. Market Trend: Ratio over time
plt.figure(figsize=(10, 6))
plt.scatter(df['SaleDate'], df['Ratio'], alpha=0.5)
plt.title('Sale/RMV Ratio over Time (Gresham)')
plt.ylabel('Sale Price / RMV')
plt.grid(True)
plt.savefig('results/figures/trend_check.png')

df = df.dropna(subset=['SaleDate', 'Ratio'])
df['DateOrdinal'] = df['SaleDate'].apply(lambda x: x.toordinal())
corr = df['DateOrdinal'].corr(df['Ratio'])
print(f"Correlation between Sale Date and Price/RMV Ratio: {corr:.3f}")

# 2. Seasonality: Ratio by Month
df['Month'] = df['SaleDate'].dt.month
monthly = df.groupby('Month')['Ratio'].median()
print("\nMedian Ratio by Month:")
print(monthly)

# Check counts per month
print("\nSales counts per month:")
print(df['Month'].value_counts().sort_index())
