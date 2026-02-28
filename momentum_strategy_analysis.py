import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load the index data ---
df = pd.read_csv("momentum.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Display data
# print(df.head())
# print(df.info())

# --- Step I: Compute momemtum score for each index ---
momentum_df = df.shift(1).rolling(window=12).mean()
momentum_df = momentum_df.dropna() # remove first 12 rows with NaN values due to rolling mean

# Display momemtum scores
# print(momentum_df.head())

# --- Step II: Rank Assets cross-sectionally ---
# Row-wise + enforce unique ranks
rank_df = momentum_df.rank(axis=1, method="first", ascending=True)

# Display ranks
# print(rank_df.head())

# --- Step III: Construct five equal weighted portfolios ---
# Align returns with ranking dates (use next month's returns)
returns_df = df.loc[rank_df.index]

# Display returns
# print(returns_df.head())

# Prepare portfolios
pf_mkt = []
pf_high = []
pf_low = []
pf_long_short = []
pf_ls_plus_mkt = []

dates = rank_df.index

for date in dates:
    ranks = rank_df.loc[date]
    returns = returns_df.loc[date]
    
    # Identify assets
    top3 = ranks.nlargest(3).index
    bottom3 = ranks.nsmallest(3).index
    all_assets = ranks.index
    
    # 1. Market portfolio (equal weight all 10)
    return_mkt = returns[all_assets].mean()
    
    # 2. High momentum (equal weight top 3)
    return_high = returns[top3].mean()
    
    # 3. Low momentum (equal weight bottom 3)
    return_low = returns[bottom3].mean()
    
    # 4. Long-short (50% long top 3, 50% short bottom 3)
    return_long_short = 0.5 * return_high - 0.5 * return_low
    
    # 5. Long-short + market
    return_ls_plus_mkt = return_mkt + return_long_short
    
    # Store results
    pf_mkt.append(return_mkt)
    pf_high.append(return_high)
    pf_low.append(return_low)
    pf_long_short.append(return_long_short)
    pf_ls_plus_mkt.append(return_ls_plus_mkt)

# Create final dataframe portfolios
portfolio_returns = pd.DataFrame({
    "pf_mkt": pf_mkt,
    "pf_high": pf_high,
    "pf_low": pf_low,
    "pf_long_short": pf_long_short,
    "pf_ls_plus_mkt": pf_ls_plus_mkt
}, index=dates)

# Display portfolios returns
# print(portfolio_returns.head())

# --- Step IV: Backtest and Evaluate Performance  ---
# Number of months per year
months_per_year = 12

# Annualized return
annualized_return = (1 + portfolio_returns.mean())**months_per_year - 1

# Annualized volatility
annualized_volatility = portfolio_returns.std() * (months_per_year**0.5)

# Sharpe ratio (risk-free rate = 0)
sharpe_ratio = annualized_return / annualized_volatility

# Cumulative returns
cumulative_returns = (1 + portfolio_returns).cumprod() - 1

# Final cumulative return (last value)
total_cumulative_return = cumulative_returns.iloc[-1]

# Combine performance metrics
performance_summary = pd.DataFrame({
    "Annual Return": annualized_return,
    "Annual Volatility": annualized_volatility,
    "Sharpe Ratio": sharpe_ratio,
    "Total Cumulative Return": total_cumulative_return # cumulative return at the end of the data period
})

# Display performance evaluation
# print("\nPerformance Summary:")
# print(performance_summary)

# Plot cumulative growth of $1 invested in each portfolio
# growth = (1 + portfolio_returns).cumprod()
# growth.plot(figsize=(12,6))
# plt.title("Growth of $1 Invested")
# plt.ylabel("Portfolio Value")
# plt.grid(True)
# plt.show()

# --- Step IV bis:  Annualized stats for individual ETFs (not portfolios) ---
# Number of months in the dataset
N = df.shape[0]
# Geometric annualized return per asset
annualized_return_assets = (df.add(1).prod())**(12 / N) - 1
annualized_vol_assets = df.std() * (months_per_year**0.5)
sharpe_ratio_assets = annualized_return_assets / annualized_vol_assets

# --- Step V: Verification ---
# Function to check calculated values against reference values
def check_values(momentum_df, rank_df, portfolio_returns, reference):
    """
    Check calculated values against reference values.
    
    reference: dict with keys
        - 'momentum': dict of {asset: value} for a given date
        - 'rank_top': list of top 3 assets
        - 'rank_bottom': list of bottom 3 assets
        - 'portfolio': dict of portfolio returns for the date
        - 'stats': dict of overall stats for a given asset
    """
    import numpy as np

    date = reference['date']
    tol = 1e-4  # tolerance for numerical comparison

    print(f"\nChecking values for {date.date()}:")

    # --- Momentum ---
    for asset, ref_value in reference['momentum'].items():
        calc_value = momentum_df.loc[date, asset]
        if np.isclose(calc_value, ref_value, atol=tol):
            print(f"Momentum {asset}: OK ({calc_value:.4%})")
        else:
            print(f"Momentum {asset}: MISMATCH! Calculated={calc_value:.4%}, Reference={ref_value:.4%}")

    # --- Rankings ---
    calc_top = list(rank_df.loc[date].nlargest(3).index)
    calc_bottom = list(rank_df.loc[date].nsmallest(3).index)

    if calc_top == reference['rank_top']:
        print(f"Top 3 assets: OK ({calc_top})")
    else:
        print(f"Top 3 assets: MISMATCH! Calculated={calc_top}, Reference={reference['rank_top']}")

    if calc_bottom == reference['rank_bottom']:
        print(f"Bottom 3 assets: OK ({calc_bottom})")
    else:
        print(f"Bottom 3 assets: MISMATCH! Calculated={calc_bottom}, Reference={reference['rank_bottom']}")

    # --- Portfolio Returns ---
    for pf, ref_value in reference['portfolio'].items():
        calc_value = portfolio_returns.loc[date, pf]
        if np.isclose(calc_value, ref_value, atol=tol):
            print(f"{pf} return: OK ({calc_value:.4%})")
        else:
            print(f"{pf} return: MISMATCH! Calculated={calc_value:.4%}, Reference={ref_value:.4%}")

    # --- Overall stats ---
    for asset, stats in reference.get('stats', {}).items():
        calc_return = annualized_return_assets[asset]
        calc_vol = annualized_vol_assets[asset]

        if np.isclose(calc_return, stats['annual_return'], atol=tol):
            print(f"{asset} annual return: OK ({calc_return:.2%})")
        else:
            print(f"{asset} annual return: MISMATCH! Calculated={calc_return:.2%}, Reference={stats['annual_return']:.2%})")

        if 'annual_vol' in stats:
            if np.isclose(calc_vol, stats['annual_vol'], atol=tol):
                print(f"{asset} annual vol: OK ({calc_vol:.2%})")
            else:
                print(f"{asset} annual vol: MISMATCH! Calculated={calc_vol:.2%}, Reference={stats['annual_vol']:.2%})")

# Reference values given
reference_march2023 = {
    'date': pd.Timestamp('2023-03-31'),
    'momentum': {'VOX': -0.0190},
    'rank_top': ['VDE', 'VIS', 'VPU'],
    'rank_bottom': ['VOX', 'VCR', 'VNQ'],
    'portfolio': {
        'pf_mkt': 0.0101,
        'pf_high': 0.0032,
        'pf_low': 0.0183,
        'pf_long_short': -0.0075,
        'pf_ls_plus_mkt': 0.0026
    },
    'stats': {
        'VOX': {'annual_return': 0.1072, 'annual_vol': 0.1640}
    }
}

# Run check
check_values(momentum_df, rank_df, portfolio_returns, reference_march2023)



