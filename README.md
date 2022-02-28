# COVID_19: Analysis, EDA, and Forecasting

## Table of Content
  * [Overview](#overview)
  * [Installation](#installation)
  * [Directory Tree](#directory-tree)
  * [Future scope of project](#future-scope)

## Overview
This project analyzes the spreading and evolution of COVID-19 around the world and forecasts the confirmed cases and fatalities using the previous observations.
The first part of the project is downloading and exploring the data set in terms of structure and features, our data cleaning method is also illustrated in this part.
The overviews of the features are given in the following section, with features selection and feature engineering.
The third part focuses on the visualization phase, where we take the cleaned and processed data and turn them into intuitive graphs.
In the fourth part, we provide the details of the selected models (Prophet, SARIMA, SARIMAX) for modeling the growth of COVID-19 cases and approaches to estimate the corresponding parameters.
Finally, export the data to .csv to use that cleaned data in tableau to make a dashboard.

Dataset from [Our World in Data](https://github.com/owid/covid-19-data/tree/master/public/data). This dataset is a collection of the COVID-19 data maintained by [Our World in Data](https://ourworldindata.org/), which is updated daily throughout the duration of the COVID-19 pandemic. 

#### Technology and tools wise this project covers:
- Python.
- Numpy and Pandas for data cleaning.
- Plotly, Seaborn, and Matplotlib for data visualization.
- Sklearn for model building.
- Jupyter notebook as IDE.
- Prophet, a procedure for forecasting time series data.
- SARIMA, a seasonal autoregressive integrated moving average model
- SARIMAX, a seasonal auto-regressive integrated moving average with eXogenous factors, is an updated version of the ARIMA model.
- Tableau, an interactive data visualization software focused on business intelligence.
- 
[![](/dashboard-covid19.png)]

## Installation
The Code is written in Python 3.9.10. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). 
## Directory Tree 
```
‎| README.md
‎| covid-19-analysis-eda-and-forecasting.ipynb
‎| covid-19-analysis-eda-and-forecasting.py
‎| dashboard-covid19.PNG
``` 

## Future Scope

* Improve Forecast Accuracy.
* Frontend: React JS, React Native