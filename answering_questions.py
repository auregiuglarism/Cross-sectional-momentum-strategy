from momentum_strategy_analysis import run_momentum_strategy_analysis
import statsmodels.api as sm

# Retrieve data
results = run_momentum_strategy_analysis()

# --- Question 1 ---
# What is the annualized return for VGT over the entire period (January 2012 - December 2025)? 
print("Annualized return for VGT:", results["annualized_return_assets"]["VGT"])

# --- Question 2 ---
# Which asset has the lowest annualized standard deviation (volatility) over the period? 
lowest_vol_asset = results["annualized_vol_assets"].idxmin()
print("Asset with lowest annualized volatility:", lowest_vol_asset)
print("Lowest annualized volatility:", results["annualized_vol_assets"][lowest_vol_asset])

# --- Question 3 ---
# What is the Sharpe ratio for VGT?
sharpe_ratio_vgt = results["sharpe_ratio_assets"]["VGT"]
print("Sharpe ratio for VGT:", sharpe_ratio_vgt)

# --- Question 4 ---
# Which asset has the highest annualized return over the entire period?
highest_return_asset = results["annualized_return_assets"].idxmax()
print("Asset with highest annualized return:", highest_return_asset)
print("Highest annualized return:", results["annualized_return_assets"][highest_return_asset])

# ------------------
# For the following questions, only consider the period from Jan 2013 - Dec 2025 
# (the “Backtesting Period”)
# ------------------

# --- Question 5 ---
# What is the average rank across all time periods for VDE? (Remember: rank 1 = lowest score, rank 10 = highest score) 
vde_rank = results["rank_df"]["VDE"]
average_rank_vde = vde_rank.mean()
print("Average rank for VDE:", average_rank_vde)

# --- Question 6 ---
# In December 2025, what was the 12-month average momentum return for the asset with rank 1 (the lowest score)?
# Find December 2025 data
dec_2025 = results["momentum_df"].index[-1]  # Last date in the dataset (should be Dec 2025)
print("December 2025 date:", dec_2025)

# Get ranks for December 2025
ranks_dec_2025 = results["rank_df"].loc[dec_2025]
print("Ranks in December 2025:", ranks_dec_2025)

# Find asset with rank 1 (lowest score)
asset_rank_1 = ranks_dec_2025.idxmin()
print("Asset with rank 1 in December 2025:", asset_rank_1)

# Get the momentum return for that asset in December 2025
momentum_return_asset_rank_1 = results["momentum_df"].loc[dec_2025, asset_rank_1]
print(f"Momentum return for {asset_rank_1} in December 2025: {momentum_return_asset_rank_1}")

# --- Question 7 ---
# In January 2013, what was the equal-weighted average return across all 10 assets that month? 
jan_2013 = results["returns_df"].index[0]  # First date in the dataset (should be Jan 2013)
print("January 2013 date:", jan_2013)

# Get returns for January 2013
returns_jan_2013 = results["returns_df"].loc[jan_2013]
print("Returns in January 2013:", returns_jan_2013)

# Calculate equal-weighted average return
equal_weighted_avg_return = returns_jan_2013.mean()
print("Equal-weighted average return for January 2013:", equal_weighted_avg_return)

# --- Question 8 ---
# Which asset has the highest average rank? 
average_ranks = results["rank_df"].mean()
highest_avg_rank_asset = average_ranks.idxmax()
print("Asset with highest average rank:", highest_avg_rank_asset)
print("Highest average rank:", average_ranks[highest_avg_rank_asset])

# --- Question 9 ---
# What is the annualized return for the high momentum equal-weighted portfolio (pf_high) over the backtest period? 
pf_high_returns = results["portfolio_returns"]["pf_high"]
months_per_year = 12
annualized_return_pf_high = (1 + pf_high_returns.mean())**months_per_year - 1
print("Annualized return for pf_high:", annualized_return_pf_high)

# --- Question 10 ---
# What is the momentum spread: the difference in annualized returns (in percentage points) between the high momentum portfolio (pf_high) and the low momentum portfolio (pf_low)? 
pf_low_returns = results["portfolio_returns"]["pf_low"]
annualized_return_pf_low = (1 + pf_low_returns.mean())**months_per_year - 1
momentum_spread = annualized_return_pf_high - annualized_return_pf_low
print("Momentum spread (pf_high - pf_low):", momentum_spread)

# --- Question 11 ---
# What is the annualized Sharpe ratio for the equal-weighted market portfolio (pf_mkt)? 
pf_mkt_returns = results["portfolio_returns"]["pf_mkt"]
annualized_return_pf_mkt = (1 + pf_mkt_returns.mean())**months_per_year - 1
annualized_vol_pf_mkt = pf_mkt_returns.std() * (months_per_year**0.5)
sharpe_ratio_pf_mkt = annualized_return_pf_mkt / annualized_vol_pf_mkt
print("Annualized Sharpe ratio for pf_mkt:", sharpe_ratio_pf_mkt)

# --- Question 12 ---
# What is the cumulative return for the high momentum portfolio (pf_high)? 
cumulative_return_pf_high = (1 + pf_high_returns).prod() - 1
print("Cumulative return for pf_high:", cumulative_return_pf_high)

# --- Question 13 ---
# What is the annualized volatility (standard deviation) of the market-neutral long-short portfolio (pf_long_short)?
pf_long_short_returns = results["portfolio_returns"]["pf_long_short"]
annualized_vol_pf_long_short = pf_long_short_returns.std() * (months_per_year**0.5)
print("Annualized volatility for pf_long_short:", annualized_vol_pf_long_short)

# --- Question 14 ---
# What is the annualized return for the long-short + market portfolio (pf_ls_plus_mkt)? 
pf_ls_plus_mkt_returns = results["portfolio_returns"]["pf_ls_plus_mkt"]
annualized_return_pf_ls_plus_mkt = (1 + pf_ls_plus_mkt_returns.mean())**months_per_year - 1
print("Annualized return for pf_ls_plus_mkt:", annualized_return_pf_ls_plus_mkt)

# --- Question 15 ---
# What is the cumulative return over the entire period for the long-short + market portfolio (pf_ls_plus_mkt)? 
cumulative_return_pf_ls_plus_mkt = (1 + pf_ls_plus_mkt_returns).prod() - 1
print("Cumulative return for pf_ls_plus_mkt:", cumulative_return_pf_ls_plus_mkt)

# --- Question 16 ---
# Regress monthly returns of pf_ls_plus_mkt against pf_mkt. What is the resulting slope (beta)?
X = pf_mkt_returns
y = pf_ls_plus_mkt_returns
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()
beta = model.params[1]
print("Beta of pf_ls_plus_mkt against pf_mkt:", beta)

# --- Question 17 ---
# What is the resulting alpha (annualized)?
alpha_monthly = model.params[0]
alpha_annualized = (1 + alpha_monthly)**months_per_year - 1
print("Alpha of pf_ls_plus_mkt against pf_mkt (annualized):", alpha_annualized)

# --- Question 18 ---
# What is the maximum drawdown of pf_high?
cumulative_returns_pf_high = (1 + pf_high_returns).cumprod()
running_max = cumulative_returns_pf_high.cummax()
drawdown = (cumulative_returns_pf_high - running_max) / running_max
max_drawdown = drawdown.min()
print("Maximum drawdown for pf_high:", max_drawdown)