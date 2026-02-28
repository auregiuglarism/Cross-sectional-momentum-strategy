import pandas as pd
import matplotlib.pyplot as plt

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
print("\nPerformance Summary:")
print(performance_summary)

# Plot cumulative growth of $1 invested in each portfolio
growth = (1 + portfolio_returns).cumprod()
growth.plot(figsize=(12,6))
plt.title("Growth of $1 Invested")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.show()



