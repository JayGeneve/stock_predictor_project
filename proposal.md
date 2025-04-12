## Research Proposal: Fitting Neural Networks to Predict Asset Price
By Joseph Carruth, Jay Geneve, Michael Jamesley, and Evan Trock
### Research Question
To address the problem of fitting various neural networks in an attempt to forecast the future price of an asset, we pose our general question as follows:

***Can a neural network effectively predict future stock prices based on historical data and other relevant market indicators?***

Some questions we specifically would like to answer include the following:

- What input features yield the best predictive power for stock price forecasting?
  - Examples include historical prices, volume, macroeconomic variables, etc.
- Does incorporating alternative data like sentiment analysis on news headlines or earnings call transcripts improve model performance?
- Can the model generate returns above the S&P 500 when used as part of a trading strategy? Risk-free Rate?
- Can we obtain a Sharpe Ratio (an equity’s risk premium over its standard deviation) or R-Squared value close to or better than the NN used in the prompt paper, "Empirical Asset Pricing via Machine Learning" ? Better than a buy-and-hold investor? 

### Needed Data
With our goal in sight, the correct accumulation of data is necessary. That being said, we need our final data set to have the observation firm-month, spanning from 1995 to the present day, and specific variables including: name (company and PERMNO), ticker, stock price, industry (SIC, NAICS, etc), volume traded (liquidity), return on S&P. As well we will need other macroeconomic time-series data like DPS, EPS, and treasury bill rate, which can be garnered from the Wharton Research Data Services (WRDS) database. Ideally, our initial information would be gathered from CRSP in WRDS, due to its clean firm-level data, which can even bolster our analysis by including delisted tickers and dividend information. We will also likely want firm accounting data, accessible from Compustat in the WRDS database. We will also aim to collect sentiment data from a headline API that will have the titles of articles from relevant news surrounding firms around specific trading days. We can then add this to the training dataset and see whether the model can understand key relationships between news headlines, sentiment, and stock returns. Our goal is to use different pattern recognition algorithms that use functions like EMA and RSI to train our model on the patterns that the value of an asset follows. We will then use data the model hasn’t seen to test its forecasting accuracy in order to iterate and improve our model over time. 
