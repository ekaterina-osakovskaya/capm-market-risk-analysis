import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

stock = "AAPL"
market = "SPY"

prices = yf.download(
    [stock, market],
    start="2020-01-01",
    auto_adjust=True,
    progress=False
)["Close"]

returns = prices.pct_change().dropna()

stock_returns = returns[stock]
market_returns = returns[market]

beta, alpha, r_value, p_value, std_err = linregress(
    market_returns,
    stock_returns
)

print("Beta:", round(beta,3))
print("Alpha:", round(alpha,6))
print("R²:", round(r_value**2,3))



plt.figure(figsize=(8,6))

plt.scatter(
    market_returns,
    stock_returns,
    alpha=0.4
)

x=np.linspace(
    market_returns.min(),
    market_returns.max(),
    100
)

plt.plot(
    x,
    alpha+beta*x,
    color="red"
)

plt.title("Security Characteristic Line")
plt.xlabel("Market Returns")
plt.ylabel("Apple Returns")
plt.grid(True)
plt.show()




risk_free=0.02

market_return=market_returns.mean()*252

expected_return=risk_free+beta*(market_return-risk_free)

print("Expected annual return:",round(expected_return,4))


