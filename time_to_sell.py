"""Analyze relationship between price and time to sell (days on market).

Data limitation: Redfin's sold-home CSV export doesn't include DOM.
Only active for-sale listings have DOM data. So we analyze:
  1. Current listings: how does asking price relate to sitting time?
  2. Current listings: does overpricing (ask vs RMV) predict longer sits?
  3. Salem: no DOM data available from county records.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def analyze():
    print("=== Price vs Time to Sell Analysis ===\n")

    print("DATA NOTE: Redfin sold-home exports do NOT include Days on Market.")
    print("           Only active for-sale listings have DOM data.")
    print("           Salem county data also has no DOM information.\n")

    # Load for-sale enriched data
    forsale = pd.read_csv(os.path.join(BASE_DIR, 'results/reports/gresham_forsale_enriched.csv'))
    forsale['DOM'] = pd.to_numeric(forsale.get('DaysOnMarket'), errors='coerce')
    valid = forsale[forsale['DOM'].notna() & (forsale['RMV'] > 0)].copy()

    print(f"Gresham for-sale listings with DOM + RMV: {len(valid)}")
    print(f"DOM range: {valid['DOM'].min():.0f} - {valid['DOM'].max():.0f} days")
    print(f"DOM median: {valid['DOM'].median():.0f} days, mean: {valid['DOM'].mean():.0f} days")

    # Key correlations
    r_price, p_price = stats.pearsonr(valid['AskingPrice'], valid['DOM'])
    r_ratio, p_ratio = stats.pearsonr(valid['AskToRMV'], valid['DOM'])
    r_rmv, p_rmv = stats.pearsonr(valid['RMV'], valid['DOM'])

    print(f"\n--- Correlations with Days on Market ---")
    print(f"  Asking price vs DOM:    r = {r_price:.3f}, p = {p_price:.4f}")
    print(f"  RMV vs DOM:             r = {r_rmv:.3f}, p = {p_rmv:.4f}")
    print(f"  Ask/RMV ratio vs DOM:   r = {r_ratio:.3f}, p = {p_ratio:.4f}")

    # Overpriced vs fairly priced
    print(f"\n--- DOM by pricing strategy ---")
    below_rmv = valid[valid['AskToRMV'] < 1.0]
    near_rmv = valid[(valid['AskToRMV'] >= 1.0) & (valid['AskToRMV'] < 1.10)]
    above_rmv = valid[valid['AskToRMV'] >= 1.10]

    groups = [
        ('Below RMV (<1.0)', below_rmv),
        ('Near RMV (1.0-1.10)', near_rmv),
        ('Above RMV (>=1.10)', above_rmv),
    ]
    print(f"  {'Pricing':<25} {'Count':>6} {'Med DOM':>9} {'Mean DOM':>10} {'Med Ask':>12}")
    print(f"  {'-'*66}")
    for label, subset in groups:
        if len(subset) > 0:
            print(f"  {label:<25} {len(subset):>6} {subset['DOM'].median():>9.0f} "
                  f"{subset['DOM'].mean():>10.0f} ${subset['AskingPrice'].median():>11,.0f}")

    # By price tier
    print(f"\n--- DOM by price tier ---")
    tiers = [
        ('Under $550K', valid[valid['AskingPrice'] < 550_000]),
        ('$550K - $700K', valid[(valid['AskingPrice'] >= 550_000) & (valid['AskingPrice'] < 700_000)]),
        ('$700K - $900K', valid[(valid['AskingPrice'] >= 700_000) & (valid['AskingPrice'] < 900_000)]),
        ('$900K+', valid[valid['AskingPrice'] >= 900_000]),
    ]
    print(f"  {'Tier':<20} {'Count':>6} {'Med DOM':>9} {'Mean DOM':>10} {'Med Ask/RMV':>12}")
    print(f"  {'-'*60}")
    for label, subset in tiers:
        if len(subset) > 0:
            print(f"  {label:<20} {len(subset):>6} {subset['DOM'].median():>9.0f} "
                  f"{subset['DOM'].mean():>10.0f} {subset['AskToRMV'].median():>12.3f}")

    # Individual listings sorted by DOM
    print(f"\n--- All listings by days on market ---")
    valid_sorted = valid.sort_values('DOM', ascending=False)
    print(f"  {'Address':<35} {'Ask':>10} {'RMV':>10} {'Ask/RMV':>8} {'DOM':>5}")
    print(f"  {'-'*72}")
    for _, row in valid_sorted.iterrows():
        print(f"  {row['Address']:<35} ${row['AskingPrice']/1000:>8,.0f}K ${row['RMV']/1000:>8,.0f}K "
              f"{row['AskToRMV']:>8.3f} {int(row['DOM']):>5}")

    # Regression: if you price at $1.1M (Viewcrest estimate), how long?
    if abs(r_price) > 0.1:
        slope_dom, int_dom, _, _, _ = stats.linregress(valid['AskingPrice'], valid['DOM'])
        pred_1100 = slope_dom * 1_100_000 + int_dom
        pred_1200 = slope_dom * 1_200_000 + int_dom
        pred_750 = slope_dom * 750_000 + int_dom
        print(f"\n--- Predicted DOM from price regression ---")
        print(f"  At $750K asking:    {max(0, pred_750):.0f} days")
        print(f"  At $1.1M asking:    {max(0, pred_1100):.0f} days")
        print(f"  At $1.2M asking:    {max(0, pred_1200):.0f} days")

    # Spearman rank correlation (better for non-linear relationships)
    rho, p_rho = stats.spearmanr(valid['AskingPrice'], valid['DOM'])
    print(f"\n  Spearman rank correlation (price vs DOM): rho = {rho:.3f}, p = {p_rho:.4f}")

    rho_ratio, p_rho_ratio = stats.spearmanr(valid['AskToRMV'], valid['DOM'])
    print(f"  Spearman rank correlation (Ask/RMV vs DOM): rho = {rho_ratio:.3f}, p = {p_rho_ratio:.4f}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Price vs Time on Market — Gresham For-Sale Listings',
                 fontsize=14, fontweight='bold')

    # 1. Asking price vs DOM
    ax = axes[0, 0]
    ax.scatter(valid['AskingPrice'] / 1000, valid['DOM'], alpha=0.6, s=40,
               color='darkorange', edgecolors='white', linewidth=0.3)
    z = np.polyfit(valid['AskingPrice'], valid['DOM'], 1)
    x_line = np.linspace(valid['AskingPrice'].min(), valid['AskingPrice'].max(), 100)
    ax.plot(x_line / 1000, np.polyval(z, x_line), 'r-', alpha=0.7,
            label=f'Trend (r={r_price:.2f})')
    ax.set_xlabel('Asking Price ($K)')
    ax.set_ylabel('Days on Market')
    ax.set_title(f'Asking Price vs DOM (n={len(valid)})')
    ax.legend(fontsize=9)

    # 2. Ask/RMV ratio vs DOM
    ax = axes[0, 1]
    ax.scatter(valid['AskToRMV'], valid['DOM'], alpha=0.6, s=40,
               color='darkorange', edgecolors='white', linewidth=0.3)
    z2 = np.polyfit(valid['AskToRMV'], valid['DOM'], 1)
    x2 = np.linspace(valid['AskToRMV'].min(), valid['AskToRMV'].max(), 100)
    ax.plot(x2, np.polyval(z2, x2), 'r-', alpha=0.7,
            label=f'Trend (r={r_ratio:.2f})')
    ax.axvline(1.0, color='green', linestyle='--', alpha=0.3, label='Ask = RMV')
    ax.set_xlabel('Asking Price / RMV Ratio')
    ax.set_ylabel('Days on Market')
    ax.set_title('Overpricing vs DOM')
    ax.legend(fontsize=9)

    # 3. DOM by pricing strategy (box plot)
    ax = axes[1, 0]
    group_data = []
    group_labels = []
    for label, subset in groups:
        if len(subset) >= 2:
            group_data.append(subset['DOM'].values)
            group_labels.append(f'{label}\n(n={len(subset)})')
    if group_data:
        bp = ax.boxplot(group_data, tick_labels=group_labels, patch_artist=True)
        colors = ['#90caf9', '#ffcc80', '#ef9a9a']
        for box, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            box.set_facecolor(color)
        ax.set_ylabel('Days on Market')
        ax.set_title('DOM by Pricing Strategy')

    # 4. DOM by price tier (box plot)
    ax = axes[1, 1]
    tier_data = []
    tier_labels = []
    for label, subset in tiers:
        if len(subset) >= 2:
            tier_data.append(subset['DOM'].values)
            tier_labels.append(f'{label}\n(n={len(subset)})')
    if tier_data:
        bp = ax.boxplot(tier_data, tick_labels=tier_labels, patch_artist=True)
        colors = ['#a5d6a7', '#66bb6a', '#388e3c', '#1b5e20']
        for box, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            box.set_facecolor(color)
        ax.set_ylabel('Days on Market')
        ax.set_title('DOM by Price Tier')

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'results/figures/time_to_sell.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {out}")


if __name__ == '__main__':
    analyze()
