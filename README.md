# kaggle_sales_forecasting
Kaggle Predict Future Sales Kudos Competition

--

## Overview
The code in this repo represents attempts to predict sales from as outlined in the Kaggle kudos competition [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview). 

--

## Raw Data
The data provided for this competition can be retrieved [here](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data) and is a sample of 3 years worth of sales - 2013 to 2015 - from the Russian software firm 1C Company. Shop, item, and category names were provided in Russian with very few English words provided for some of item descriptions.

--

## Modules
Module ```sales_data``` contains the ```SalesData``` class that merges the sales, item, and category information into Pandas dataframes. Two key objects that are instantiated from this class are ```SalesData.daily_sales``` and ```SalesData.monthly_sales``` to facilitate EDA of the datasets.

Module ```nlp``` implements traditional natural language processes, such as tokenization and CountVecorizer, to break out categorical data from the shop names. The Russian language was left intact, but peripheral analysis of the names revealed words such as "Mega" and "Mall", as well as city names in which the shops are located.

Module ```lstm``` contains the ```ModelInput``` class - which prepares the data - and the ```LSTMModel``` class - which builds and trains the LSTM model to ultimately make the predictions.