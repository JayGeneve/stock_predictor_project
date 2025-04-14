# Research Proposal: Fitting Neural Networks to Predict Asset Price
## By Joseph Carruth, Jay Geneve, Michael Jamesley, and Evan Trock
# Research Question
To address the problem of fitting various neural networks in an attempt to forecast the future price of an asset, we pose our general question as follows:

Can a neural network effectively predict future stock prices based on historical data and other relevant market indicators?

Some questions we specifically would like to answer include the following:

What input features yield the best predictive power for stock price forecasting?
Examples include historical prices, volume, macroeconomic variables, etc.
Can the model generate returns above the S&P 500 when used as part of a trading strategy? Risk-free Rate?
Can we obtain a Sharpe Ratio (an equity’s risk premium over its standard deviation) or R-Squared value close to or better than the NN used in the prompt paper, "Empirical Asset Pricing via Machine Learning" ? Better than a buy-and-hold investor? 
# Needed Data
Our final data set will be firm-month observations, ideally spanning from 1995 to the present day. The dataset will include information about the company (e.g., name, PERMNO, ticker, and industry), volume traded (liquidity), and monthly return. We hope to get firm returns from Wharton Research Data Services (WRDS), because its returns include dividends and delisting information.

Additionally, we will collect a suite of predictor variables. This will include but not be limited to
S&P returns and other, macroeconomic time-series data like DPS, EPS, and treasury bill rate (which can be collected via FRED)
Firm accounting data, via Compustat 
Known asset pricing signals (i.e.possible predictors) via the Open Asset Pricing dataset

We will also likely want firm accounting data, accessible from Compustat in the WRDS database. We will also aim to collect sentiment data from a headline API that will have the titles of articles from relevant news surrounding firms around specific trading days. 

Using these signals, we will determine whether we can build a model that can understand key relationships between news headlines, sentiment, and stock returns. 

Our goal is to use different pattern recognition algorithms that use functions like EMA and RSI to train our model on the patterns that the value of an asset follows. We will then use data the model hasn’t seen to test its forecasting accuracy in order to iterate and improve our model over time. 

# Resources
Data (October 2024 Release) – Open Source Asset Pricing This dataset contains two kinds of data: 
Portfolio returns, where the portfolio are the returns earned by trading long minus short “anomaly” portfolios. This isn’t relevant to our project.
Stock-level Signals. This is the data we need. They have a point and click version that contains most signals, except a few that require WRDS access.
9.7. Open Asset Pricing This includes a walkthrough of obtaining and downloading data on “anomaly portfolio returns”. More importantly, it includes three examples we need to understand:
A quick tour
How to combine it with stock price returns from CRSP. The key point here is to lag the predictor variables, so we use, e.g., January data to predict February returns.
Machine Learning Example. 
5.4. Coding in Teams - Branching on our work repo will be very useful, so we can try different experiments. 
5.5. Sharing large files - We can all download the stock-level signal file individually, or share it this way. The former is probably easier. 


# Process
The goal of this project is to upgrade the final part of the Machine Learning Example above: To try many kinds of models, many different set ups of models, and various sets of predictor variables.
Data we will construct:


## Dataset #1: 
We should save, for each model, all of the stock level predictions (which then are used to sort stocks into portfolios). We can then use this to see what kinds of stocks a given strategy recommends buying on the long side and what it recommends shorting. This is an important check, because some “anomalies” require you to short small firms that are very hard to short.
Think of this as building a dataset like the midterm. For a PERMNO-month (row), a given column refers to the model that made it, and values are the prediction the model made for that firm-month. We can make it so that this dataset can be saved, reloaded, and built up as we go, because this project will not end up with code you run in one go. For this, Prof. Bowen’s midterm answer code has features we can use! (Load data if it exists, only do a model if we don’t have predictions for it already, etc.)
Dataset #2: Contains the monthly performance for each model in Dataset #1. This is actually the more important dataset.
For a given model and month, we take Dataset #1 and sort the stocks into 5 buckets (“portfolios “), and get the average return. The long-short return for a model-month is the return of the 5th bucket minus the 1st bucket.


# Outputs
Display our results on a dashboard

## From Dataset #2:
Tables and figures that show how different models compare in terms of out of sample performance. 
Using these, we can assess what kinds of model choices increase and decrease our out-of-sample performance? We can compare this to Gu, Kelly & Xiu (2020). 
A plot of the cumulative returns to each possible model (the long-short portfolio), as though we traded it out of sample from 2000-2024.
We will pick our favorite model, and then use the stock-level predictions to make the Canonical Asset Pricing Table 1. 
Finally, we will try to upload our stock prediction signals (Dataset #1) to Assaying Anomalies and get a report back. 



## Bibliography
Gu, Shihao and Kelly, Bryan T. and Xiu, Dacheng, Empirical Asset Pricing via Machine Learning (September 13, 2019). Chicago Booth Research Paper No. 18-04, 31st Australasian Finance and Banking Conference 2018, Yale ICF Working Paper No. 2018-09, Available at SSRN: https://ssrn.com/abstract=3159577 or http://dx.doi.org/10.2139/ssrn.3159577
