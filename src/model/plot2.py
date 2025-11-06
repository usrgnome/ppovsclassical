# --- requirements: pandas, matplotlib ---
import io
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1) Load data
#    EITHER: point to 'results.csv'
#    OR: paste your raw CSV into RAW below and set USE_RAW=True
# =========================================================
USE_RAW = False  # set to True to use the RAW string block below
CSV_PATH = "eval_daily_results.csv"

RAW = r"""
policy,episode,day,action_price,action_restock_frac,price,opening_inventory,restocked,expired,visitors,sales,unmet_demand,closing_inventory,revenue,restock_cost,holding_cost,day_profit,cum_profit,waste_units,waste_value_at_cost,missed_sales_units,missed_revenue,missed_margin,cum_waste_units,cum_missed_sales_units,stockout,capacity,capacity_left_opening
<PASTE-YOUR-CSV-ROWS-HERE>
"""

def load_data():
    if USE_RAW:
        df = pd.read_csv(io.StringIO(textwrap.dedent(RAW).strip()))
    else:
        df = pd.read_csv(CSV_PATH)
    # enforce numeric where appropriate (robust to scientific or float strings)
    numeric_cols = [
        'episode','day','action_price','action_restock_frac','price','opening_inventory',
        'restocked','expired','visitors','sales','unmet_demand','closing_inventory',
        'revenue','restock_cost','holding_cost','day_profit','cum_profit','waste_units',
        'waste_value_at_cost','missed_sales_units','missed_revenue','missed_margin',
        'cum_waste_units','cum_missed_sales_units','stockout','capacity','capacity_left_opening'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['policy'] = df['policy'].astype(str)
    return df

df = load_data()

# =========================================================
# 2) Helper features
# =========================================================
# service level (fill rate) = 1 - unmet_demand/visitors (guard 0 division)
df['service_level'] = 1 - (df['unmet_demand'].fillna(0) / df['visitors'].replace(0, np.nan))
df['service_level'] = df['service_level'].fillna(1.0)

# conversion = sales / visitors
df['conversion'] = (df['sales'] / df['visitors'].replace(0, np.nan)).fillna(0.0)

# average realized margin per unit sold (rough proxy)
# margin proxied via day_profit / max(sales,1) to avoid div/0
df['profit_per_sale'] = df['day_profit'] / df['sales'].replace(0, np.nan)
df['profit_per_sale'] = df['profit_per_sale'].fillna(0.0)

# restock indicator
df['restock_flag'] = (df['restocked'] > 0).astype(int)

# capacity slack at opening (share of capacity left unused)
df['opening_utilization'] = 1 - (df['capacity_left_opening'] / df['capacity'])

# waste rate vs available units (opening + restocked)
avail_units = (df['opening_inventory'].fillna(0) + df['restocked'].fillna(0)).replace(0, np.nan)
df['waste_rate'] = (df['expired'] / avail_units).fillna(0.0)

# missed sales rate vs demand (sales + unmet_demand)
demand = df['sales'].fillna(0) + df['unmet_demand'].fillna(0)
df['miss_rate'] = (df['unmet_demand'] / demand.replace(0, np.nan)).fillna(0.0)

# price dispersion (per policy later)
# profit volatility (std of day_profit) will be computed per group

# =========================================================
# 3) Policy-level headline KPIs (RAW ONLY — no benchmark deltas)
# =========================================================
agg_map = {
    'revenue': 'sum',
    'day_profit': ['sum','mean','std'],
    'visitors': 'sum',
    'sales': 'sum',
    'unmet_demand': 'sum',
    'missed_revenue': 'sum',
    'missed_margin': 'sum',
    'waste_units': 'sum',
    'waste_value_at_cost': 'sum',
    'stockout': 'sum',
    'restock_flag': 'mean',
    'service_level': 'mean',
    'conversion': 'mean',
    'profit_per_sale': 'mean',
    'price': ['mean','std'],
    'opening_utilization': 'mean',
    'waste_rate': 'mean',
    'miss_rate': 'mean'
}
pol = df.groupby('policy').agg(agg_map)

# flatten columns
pol.columns = ['_'.join([c for c in col if c]).strip('_') for col in pol.columns.values]
pol = pol.rename(columns={
    'day_profit_sum':'total_profit',
    'day_profit_mean':'avg_day_profit',
    'day_profit_std':'profit_std',
    'restock_flag_mean':'restock_days_share',
    'price_mean':'avg_price',
    'price_std':'price_std',
    'opening_utilization_mean':'avg_opening_utilization',
    'service_level_mean':'avg_service_level',
    'conversion_mean':'avg_conversion',
    'profit_per_sale_mean':'avg_profit_per_sale',
    'waste_rate_mean':'avg_waste_rate',
    'miss_rate_mean':'avg_miss_rate',
    'revenue_sum':'revenue_sum',
    'visitors_sum':'visitors_sum',
    'sales_sum':'sales_sum',
    'unmet_demand_sum':'unmet_demand_sum',
    'missed_revenue_sum':'missed_revenue_sum',
    'missed_margin_sum':'missed_margin_sum',
    'waste_units_sum':'waste_units_sum',
    'waste_value_at_cost_sum':'waste_value_at_cost_sum',
    'stockout_sum':'stockout_sum'
})

# derived rates (raw)
pol['profit_margin_on_revenue'] = pol['total_profit'] / pol['revenue_sum'].replace(0,np.nan)
pol['waste_units_per_100_sales'] = 100 * pol['waste_units_sum'] / pol['sales_sum'].replace(0,np.nan)
pol['missed_sales_share'] = pol['unmet_demand_sum'] / (pol['sales_sum'] + pol['unmet_demand_sum']).replace(0,np.nan)

# tidy sort by total_profit (descending)
pol = pol.sort_values('total_profit', ascending=False)

# keep a clear, raw summary view
summary_raw = pol[[
    'total_profit','revenue_sum','avg_day_profit','profit_std',
    'visitors_sum','sales_sum','unmet_demand_sum',
    'missed_revenue_sum','missed_margin_sum',
    'waste_units_sum','waste_value_at_cost_sum',
    'stockout_sum',
    'restock_days_share','avg_opening_utilization',
    'avg_service_level','avg_conversion','avg_profit_per_sale',
    'avg_price','price_std',
    'avg_waste_rate','avg_miss_rate',
    'waste_units_per_100_sales','missed_sales_share',
    'profit_margin_on_revenue'
]].copy()

# =========================================================
# 4) Episode leaders/laggards (per policy) — RAW ONLY
# =========================================================
episode_agg = df.groupby(['policy','episode']).agg(
    total_profit=('day_profit','sum'),
    total_revenue=('revenue','sum'),
    service_level=('service_level','mean'),
    conversion=('conversion','mean'),
    waste_units=('waste_units','sum'),
    unmet_demand=('unmet_demand','sum'),
    stockouts=('stockout','sum'),
)

# one row per policy (policy, episode)
idx_best = episode_agg.groupby('policy')['total_profit'].idxmax()
idx_worst = episode_agg.groupby('policy')['total_profit'].idxmin()
best_eps = episode_agg.loc[idx_best]
worst_eps = episode_agg.loc[idx_worst]

# =========================================================
# 5) Print narrative-style interpretations (RAW ONLY)
#     (Fixed: avoid DataFrame .name error for MultiIndex)
# =========================================================
pd.options.display.float_format = '{:,.3f}'.format

print("\n=== HEADLINE: Policy performance (RAW, sorted by TOTAL PROFIT) ===\n")
print(summary_raw)

print("\n=== STABILITY / VOLATILITY (RAW) ===")
vol = pol[['profit_std','price_std']].sort_values('profit_std')
for pol_name, row in vol.iterrows():
    print(f"• {pol_name}: profit volatility (std)={row['profit_std']:,.2f}; "
          f"price dispersion (std)={row['price_std']:,.2f}")

print("\n=== OPERATIONAL BEHAVIOR (RAW) ===")
ops = pol[['restock_days_share','avg_opening_utilization','avg_waste_rate','avg_miss_rate','stockout_sum']].copy()
ops = ops.sort_values('restock_days_share', ascending=False)
for pol_name, row in ops.iterrows():
    print(f"• {pol_name}: restock on {row['restock_days_share']:.1%} of days; "
          f"opening utilization={row['avg_opening_utilization']:.1%}; "
          f"waste rate={row['avg_waste_rate']:.1%}; miss rate={row['avg_miss_rate']:.1%}; "
          f"stockouts={int(row['stockout_sum'])}.")

print("\n=== BEST/WORST EPISODES BY POLICY (RAW, by total_profit) ===")
# get the exact MultiIndex labels for best/worst (one per policy)
best_idx = episode_agg.groupby(level=0)['total_profit'].idxmax()
worst_idx = episode_agg.groupby(level=0)['total_profit'].idxmin()

for pol_name in best_idx.index:
    be_label = best_idx.loc[pol_name]      # ('policy', best_episode)
    we_label = worst_idx.loc[pol_name]     # ('policy', worst_episode)

    be = episode_agg.loc[be_label]         # Series (single row)
    we = episode_agg.loc[we_label]         # Series (single row)

    be_ep = int(be_label[1])
    we_ep = int(we_label[1])

    print(f"\n• {pol_name}")
    print(f"   - BEST  ep {be_ep}: profit={be['total_profit']:,.2f}, "
          f"service={be['service_level']:.1%}, waste_units={be['waste_units']:,.0f}, "
          f"unmet={be['unmet_demand']:,.0f}, stockouts={int(be['stockouts'])}")
    print(f"   - WORST ep {we_ep}: profit={we['total_profit']:,.2f}, "
          f"service={we['service_level']:.1%}, waste_units={we['waste_units']:,.0f}, "
          f"unmet={we['unmet_demand']:,.0f}, stockouts={int(we['stockouts'])}")

# =========================================================
# 6) Quick plots for slide-ready visuals (unchanged visuals, but raw comparisons)
# =========================================================
plt.figure()
pp = df.groupby(['policy','day'])['day_profit'].mean().unstack(0)
pp.plot()
plt.title("Average Day Profit by Policy")
plt.xlabel("Day")
plt.ylabel("Profit")
plt.legend(title="Policy")
plt.tight_layout()
plt.show()

plt.figure()
sl = df.groupby('policy')['service_level'].mean().sort_values(ascending=False)
sl.plot(kind='bar')
plt.title("Average Service Level by Policy")
plt.ylabel("Service Level")
plt.tight_layout()
plt.show()

plt.figure()
trade = df.groupby('policy').agg(
    avg_waste_rate=('waste_rate','mean'),
    avg_miss_rate=('miss_rate','mean'),
    total_profit=('day_profit','sum')
)
plt.scatter(trade['avg_miss_rate'], trade['avg_waste_rate'],
            s=(trade['total_profit']-trade['total_profit'].min()+1)
/ trade['total_profit'].mean()*80)
for name, (x,y,_) in trade.iterrows():
    plt.annotate(name, (x,y), xytext=(5,5), textcoords='offset points')
plt.xlabel("Missed-Demand Rate")
plt.ylabel("Waste Rate")
plt.title("Waste vs Missed-Demand Trade-off (bubble ~ profit)")
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(pol['price_std'], pol['total_profit'])
for name, (x, y) in pol[['price_std','total_profit']].iterrows():
    plt.annotate(name, (x,y), xytext=(5,5), textcoords='offset points')
plt.xlabel("Price Std (Volatility)")
plt.ylabel("Total Profit")
plt.title("Price Volatility vs Profit")
plt.tight_layout()
plt.show()
