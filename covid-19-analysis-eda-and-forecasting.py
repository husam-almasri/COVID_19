#!/usr/bin/env python
# coding: utf-8

# Loading of Notebook might take some time because of Plotly visualizations. Kindly be patient!

# #### The objective of the Notebook:
# 
# The objective of this notebook is to study the COVID-19 outbreak with the help of some basic visualizations techniques. Perform predictions and Time Series forecasting in order to study the impact and spread of the COVID-19 in the coming days.
# 
# In this notebook I will try to use Seasonal ARIMA Models, Seasonal ARIMA Models with eXogenous factors, and FBProphet.

# In[1]:


# Install GitPython module to update the project dataset automatically.
get_ipython().system('pip install gitpython')
# Time series Stationary check, estimate statistical models, and perform statistical tests.
get_ipython().system(' pip install statsmodels')
# A statistical library designed to fill the void in Python's time series analysis capabilities.
get_ipython().system(' pip install pmdarima')
# Library for time series distances.
get_ipython().system(' pip install dtaidistance')


# Fitch the last updated data from 'ourworldindata.org', rename the csv file by date

# In[2]:


import urllib.request
from datetime import datetime, timedelta
try:
    print("Downloading the file ... ")
    timenow = datetime.now()
    #timenow_iso = timenow.strftime('%Y_%m_%dT%H_%M_%S') #Time ISO to second resolution
    timenow_iso = timenow.strftime('%Y_%m_%d') #Time ISO to day resolution
    url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv' #url
    covid= f'covid_{timenow_iso}.csv' # output's file name/ location
    urllib.request.urlretrieve(url, covid)
    print(f"File {covid} is saved!")
except Exception as e:
    print("Downloading file error: " + str(e))


# Import libraries

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
get_ipython().run_line_magic('matplotlib', 'inline')
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# Display all the columns and rows of dataframe
pd.pandas.set_option('display.max_columns',None)
pd.pandas.set_option('display.max_rows',None)


# ## The lifecycle of the project
# 1. Data Analysis
# 2. Feature Selection
# 3. Feature Engineering
# 4. Data Visualization
# 5. Model Building

# ## Data Analysis Phase
# #### To understand more about:
# 1. Explore missing values.
# 2. All the numerical and categorical variables.
# 3. Correlation among features.

# In[5]:


# Load data to DataFrame
df1=pd.read_csv(covid)
df1.head()


# In[6]:


df1.info()


# Dataset has 67 columns with 157476 entries, except (iso_code,location, and date), all features have missing values.

# In[7]:


# Function return the missing values percentage per feature
def miss_per_col(df):
    cols_na=df.columns[df.isnull().any()]
    # The percentage of missing values for eah feature
    for col in cols_na:
        print(col,df[col].count(),df[col].isnull().sum(),(np.round(df[col].isnull().mean()*100)),'%')
miss_per_col(df1)


# In[8]:


# Function return the missing values percentage per feature per any country has a missing value 
def miss_per_country_per_col(df):
    cols_na=df.columns[df.isnull().any()]
    print(cols_na)
    # The percentage of missing values for each feature by country
    for row in df['location'].unique():
        for col in cols_na:
            x=df[df['location']==row][col].count()
            if x==0:
                print(row,col,(x))


# In[9]:


# miss_per_country_per_col(df1)


# Explore the missing values in the continent feature

# In[10]:


# Check continent feature
df1[df1['continent'].isnull()].groupby('location')['iso_code'].count()


# The continent values in the location feature is same values in continent feature, so will drop them.
# There is four values of location deviding the countries depending on the income, but it relys on data from 2009, so will delete them also.

# Show the first day has cases records 

# In[11]:


df1.groupby('date')[['new_cases','total_cases']].sum().head(25)


# ## Features Selection
# #### To reduce the features and  skip the not useful ones.

# In[12]:


# Drop the unuseful features, either can get their values from other features or most of their values are null.
df2=df1.copy()
df2=df2.drop(columns=['new_cases_smoothed', 'new_deaths_smoothed','total_cases_per_million',
       'new_cases_per_million', 'new_cases_smoothed_per_million',
       'total_deaths_per_million', 'new_deaths_per_million',
       'new_deaths_smoothed_per_million', 'icu_patients',
       'icu_patients_per_million', 'hosp_patients',
       'hosp_patients_per_million', 'weekly_icu_admissions',
       'weekly_icu_admissions_per_million', 'weekly_hosp_admissions',
       'weekly_hosp_admissions_per_million', 'total_tests_per_thousand',
       'new_tests_per_thousand', 'new_tests_smoothed',
       'new_tests_smoothed_per_thousand', 'tests_per_case', 'tests_units',
       'total_boosters', 'new_vaccinations_smoothed',
       'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
       'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred',
       'new_vaccinations_smoothed_per_million',
       'new_people_vaccinated_smoothed',
       'new_people_vaccinated_smoothed_per_hundred', 'aged_70_older',
        'extreme_poverty', 'female_smokers', 'male_smokers',
       'handwashing_facilities', 'hospital_beds_per_thousand',
       'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
       'excess_mortality', 'excess_mortality_cumulative_per_million'])


# ## Feature Engineering Phase
# #### Performing:
# 1. Drop the missing values in categorical feature (continent).
# 2. Fill the missing values in numerical features.
# 3. Temporal feature.

# Drop the continent and any observation other than a country from location feature
# 

# In[13]:


df3=df2.copy()
df3=df3[df3['continent'].notna()].reset_index(drop=True)


# Drop the days that have no records, which are from (2020-01-01 to 2020-01-21)	

# In[14]:


df3=df3[(df3['date'])>('2020-01-21')].reset_index(drop=True)


# In[15]:


# Check the first record day 
df3.groupby('date').sum()[['new_cases','total_cases']].head(1)


# To fill null values in total features, we should consider that, for any record day,  should be greater or equal the previuos record day in its feature.
# If the first days of a total feature is null, will fill them with zero.

# In[16]:


# Fill the Null values in all total features
df4=df3.copy()
df4[['total_cases','total_deaths','total_tests','total_vaccinations','people_fully_vaccinated','stringency_index']] = df3.groupby(['location'], sort=False)[['total_cases','total_deaths','total_tests','total_vaccinations','people_fully_vaccinated','stringency_index']].apply(lambda x: x.ffill().fillna(0))


# Fill missing values of ('reproduction_rate','positive_rate') by zero

# In[17]:


# Fill the Null values by zero
df4[['reproduction_rate','positive_rate']]=df3[['reproduction_rate','positive_rate']].fillna(0)


# To fill null values in new features, we should consider that all records should be compatable wihh its total features.

# In[18]:


# Finction fills the Null values in all new features with respect to its total features
def fillna_feature(df,new,total):
    x=df[df.groupby('location')[total].diff()<0].index
    while len(x)>0:
        for i in x:
            df.loc[i,total]=df.loc[i-1,total]
        x=df[df.groupby('location')[total].diff()<0].index
    df[new]=df.groupby('location')[total].diff()
    df.loc[df.groupby('location')[new].head(1).index, new] = df.loc[df.groupby('location')[new].head(1).index, total]
    return df


# In[19]:


df5=df4.copy()


# In[20]:


# Update the new_cases feature
df5=fillna_feature(df5,'new_cases','total_cases')


# In[21]:


# Update the new_deaths feature
df5=fillna_feature(df5,'new_deaths','total_deaths')


# In[22]:


# Update the new_tests feature
df5=fillna_feature(df5,'new_tests','total_tests')


# In[23]:


# Update the new_vaccinations feature
df5=fillna_feature(df5,'new_vaccinations','total_vaccinations')


# In[24]:


# Change the people_vaccinated and people_fully_vaccinated feature names to be more clear
df6=df5.copy()
df6.rename(columns={'people_vaccinated':'total_first_shot_vac','people_fully_vaccinated':'total_fully_vaccinated'}, inplace=True)
df6.head()


# In[25]:


# Update the total_first_shot_vac feature by the subtract total_fully_vaccinated from total_vaccinations
df7=df6.copy()
for i in range(0,len(df7)):
    df7.loc[i,'total_first_shot_vac']=df7.loc[i,'total_vaccinations']-df7.loc[i,'total_fully_vaccinated']


# In[26]:


# Check the missing values percentage per feature
miss_per_col(df7)


# In[27]:


# Chart for all features have any missing value 
def missing_values_chart(df):
    missing_values = df.isnull().sum() / len(df)
    missing_values = missing_values[missing_values > 0]
    missing_values.sort_values(inplace=True)
    missing_values = missing_values.to_frame()
    missing_values.columns = ['Missed_values']
    missing_values.index.names = ['Feature']
    missing_values['Feature'] = missing_values.index
    sns.set(style="whitegrid", color_codes=True)
    sns.barplot(x = 'Feature', y = 'Missed_values', data=missing_values)
    plt.xticks(rotation = 90)
    plt.show()


# Drop countries that has a lot of missing values in most of features.

# In[28]:


# List of countries with missed data 
def countries_with_missed_data(df):
    countries_with_missed_data_list=[]
    # The percentage of missing values for eah feature by country
    for row in df['location'].unique():
        for col in df.columns[df.isnull().any()]:
            x=df[df['location']==row][col].count()
            if x==0:
                countries_with_missed_data_list.append(row)
                break
    return countries_with_missed_data_list


# In[29]:


countries_with_missed_data_list=countries_with_missed_data(df7)
# countries_with_missed_data_list


# In[30]:


# Except these countries from the countries_with_missed_data_list
countries=["Syria","Taiwan","Somalia","Hong Kong","Cuba",'South Sudan']
for i in countries:
    countries_with_missed_data_list.remove(i)
    print(i)


# In[31]:


# Drop countries_with_missed_data_list from location feature
df8=df7.copy()
df8=df8[~df8['location'].isin(countries_with_missed_data_list)].reset_index(drop=True)


# In[32]:


# Check the countries that have any missing values
# miss_per_country_per_col(df8)


# In[33]:


df8.info()


# In[34]:


df8.dtypes


# Convert date feature type to Date.

# In[35]:


df9=df8.copy()
df9['date']=pd.to_datetime(df9['date'])
df9.dtypes


# In[36]:


# Show the countries that have any missing values in population_density feature
df9[df9['population_density'].isnull()].groupby('location').sum()


# In[37]:


# Function fill the missing values in population_density feature with respect to its population and square_km
def population_density(df,location,km_2):
    ind=df[df['location']==location].index
    df.loc[ind[0]:ind[-1],"population_density"].fillna(value=np.round((df[df['location'] == location].loc[:, 'population']/km_2),decimals=3),inplace=True)


# In[38]:


# Fill the missing values in population_density feature with respect to its population and square_km
df10=df9.copy()
population_density(df10,'South Sudan',644329)
population_density(df10,'Syria',185180)
population_density(df10,'Taiwan',36197)
df10.groupby('location')['population_density'].max()


# In[39]:


# Check the features that have any missing values
missing_values_chart(df10)


# In[40]:


# Function fill the missing values in the given feature by a given value
def fillna_features(df,location,feature,value):
    ind=df[df['location']==location].index
    df.loc[ind[0]:ind[-1],feature].fillna(value=value,inplace=True)


# In[41]:


# Fill the missing values in the given feature by a given value
df11=df10.copy()
fillna_features(df11,'Syria','diabetes_prevalence',9.9)
fillna_features(df11,'Taiwan','diabetes_prevalence',8.3)
# https://www.worldometers.info/gdp/gdp-per-capita/
fillna_features(df11,'Cuba','gdp_per_capita',8541)
fillna_features(df11,'Syria','gdp_per_capita',2030)
fillna_features(df11,'Taiwan','gdp_per_capita',32787)
fillna_features(df11,'Somalia','gdp_per_capita',309)
# https://www.socialindicators.org.hk/en/indicators/health/6.15
fillna_features(df11,'Hong Kong','cardiovasc_death_rate',52.32)
fillna_features(df11,'Somalia','human_development_index',0.285)
fillna_features(df11,'Taiwan','human_development_index',0.916)
fillna_features(df11,'Taiwan','aged_65_older',16)
fillna_features(df11,'Syria','aged_65_older',4)


# Cast all total and new features to int

# In[42]:


float_feature_list=['total_cases','total_deaths','total_tests',
                    'total_vaccinations','new_cases','new_deaths',
                    'new_tests','new_vaccinations','total_fully_vaccinated',
                    'total_first_shot_vac','population']
for feature in float_feature_list:
    df11[feature]=df11[feature].astype('int32')


# In[43]:


# Save the cleaned dataset in CSV file
df11.to_csv('covid_19_cleaned.csv', index=False)


# Get the total deaths, tests, cases, and vaccinated for each continent

# In[44]:


df11.groupby(['location'])[['continent','total_deaths','total_tests','total_cases','total_fully_vaccinated','total_first_shot_vac']].last().groupby('continent').sum()


# Final Check for cleaned data

# In[45]:


df11.info()


# ## Data Visualization

# Plot min/max 20 countries in a total/new (cases, deathes, vac or tests) worldwide

# In[46]:


# Plot function to show min_max selected numder of countries in a total/new (cases, deathes, vac or tests) worldwide
def worldwide(df,min_max,total_new,n=20):
# df->dataframe
# min_max->the minimum or maximum countries
# n->number of countries to show
# total_new-> total/new (cases, deathes, vac or tests) 
    country_plot = df.groupby('location')[total_new].max().reset_index().sort_values(total_new,ascending=False)
    if min_max=='max':
        fig =px.bar(country_plot.head(n),x='location',y=total_new,template='none',title=f'{total_new} worldwide')
    else:
        fig =px.bar(country_plot.tail(n),x='location',y=total_new,template='none',title=f'{total_new} worldwide')

    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.show()


# In[47]:


worldwide(df11,'max','total_tests')


# Summary Plot of Worldwide  confirmed cases, deaths, vaccinated, and tested

# In[48]:


df11.groupby(['location'])[['continent','total_deaths','total_tests','total_cases','total_fully_vaccinated']].last().groupby('continent').sum()


# Plot a chart of reproduction rate per country

# In[49]:


for i in df11['location'].unique():
    df11[df11['location']==i].groupby('date')['reproduction_rate'].max().plot()
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.ylabel('Reproduction rate')
    plt.title(i)
    plt.show()


# Plot a chart of stringency index per country

# In[50]:


for i in df11['location'].unique():
    df11[df11['location']==i].groupby('date')['stringency_index'].max().plot()
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.ylabel('Stringency index')
    plt.title(i)
    plt.show()


# Plot a chart of reproduction rate per continent

# In[51]:


for i in df11['continent'].unique():
    df11[df11['continent']==i].groupby('date')['reproduction_rate'].mean().plot()
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.ylabel('Reproduction rate')
    plt.title(i)
    plt.show()


# Plot a chart of stringency index per continent

# In[52]:


for i in df11['continent'].unique():
    df11[df11['continent']==i].groupby('date')['stringency_index'].mean().plot()
    plt.xlabel(i)
    plt.xticks(rotation=45)
    plt.ylabel('Stringency index')
    plt.title(i)
    plt.show()


# Plot min_max 10 countries in a total/new (cases, deathes, vac or tests)for a continent

# In[53]:


# Plot function to show min_max countries in a total/new (cases, deathes, vac or tests) for a continent
def min_max(df,min_max,n,con,total_new,table):
# df->dataframe
# min_max->the minimum or maximum countries
# n->number of countries to show
# con->continent
# total_new-> total/new (cases, deathes, vac or tests) 
# table->show a table of the countries
    temp_df=df[df['continent']==con].groupby('location')[total_new].last()
    if (n>(len(temp_df)))|(n<=0):
        n=len(temp_df)
    if min_max=='max':
        temp_df=temp_df.nlargest(n=n)
    else:
        temp_df=temp_df.nsmallest(n=n)
    temp_df.plot.bar()
    plt.xlabel("Country")
    plt.xticks(rotation=90)
    plt.ylabel(total_new)
    plt.title(f"{min_max} {n} countries in {con} by {total_new}")
    plt.show()
    if table==True:
         return temp_df.head(n)


# In[54]:


min_max(df11,"min",8,"Africa",'total_deaths',True)


# Plot new features by continent for the whole period of dataset

# In[55]:


new=['new_cases','new_deaths','new_tests','new_vaccinations']
for show in new: 
    i=0
    data=[]
    color=['green','blue','red','black','brown','orange']
    for con in df11.groupby('continent')['continent'].unique():

        globals()[f"trace{i}"] = go.Scatter(
                            x = df11[(df11['continent']==con[0])].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                            y = df11[(df11['continent']==con[0])].groupby(['date','continent'])[show].sum(),
                            mode = "lines",
                            name = con[0],
                            marker = dict(color = color[i]),
        )
    #     globals()[f"trace{i}"]=trace1
        data.append(globals()[f"trace{i}"])
        i+=1

    data = [trace1,trace2,trace3,trace4,trace5]
    layout = dict(title = show,
                  xaxis= dict(title= f'# {show} day by day',ticklen= 5,zeroline= False)
                 )
    fig = dict(data = data, layout = layout)
    iplot(fig)


# Plot new deaths, new cases, and non-Pharmaceutical features clusters over countries

# In[56]:


non_Pharmaceutical_list=['reproduction_rate','stringency_index','positive_rate','population_density',
         'median_age','aged_65_older','gdp_per_capita','cardiovasc_death_rate',
         'diabetes_prevalence','life_expectancy','human_development_index']
for feature in non_Pharmaceutical_list:
    df_grouped = df11.groupby(['location','continent']).agg(
        {'new_deaths': np.sum, feature: np.mean, 'new_cases':np.sum}
    ).reset_index()
    fig = px.scatter(df_grouped, 
                     x="new_deaths", y=feature, size="new_cases",
                     color="continent",hover_name="location", log_x=True,
                     size_max=60,title = f'New deaths, new cases, and {feature} clusters over countries')
    fig.show()


# Plot clusters for new deaths, new cases, and one/double shot/s vaccinated features over countries

# In[57]:


one_double_shot=['total_first_shot_vac','total_fully_vaccinated']
for feature in one_double_shot:
    df_grouped = df11.groupby(['location','continent']).agg(
        {'new_deaths': np.sum, feature: lambda x: x.iloc[-1], 'new_cases':np.sum}
    ).reset_index()
    fig = px.scatter(df_grouped, 
                     x="new_deaths", y=feature, size="new_cases",
                     color="continent",hover_name="location", log_x=True,
                     size_max=60,title = f'New deaths, new cases, and {feature} clusters over countries')
    fig.show()


# Plot pie chart of total or new feature around the world

# In[58]:


def pie_world(df,total_new):
    df_temp = df.groupby(['continent','location'])[total_new].max().reset_index()
    fig = px.sunburst(df_temp, path=['continent','location'], values=total_new,
                      title= f'{total_new} around the world',
                      height=620,template='none')
    fig.show()
pie_world(df11,'total_deaths')


# Correlation Analysis

# Plot the correlation between a feature and all others

# In[59]:


def feature_cor(feature):
    correlations = df11.corr()[feature].abs().sort_values(ascending=False).drop(feature,axis=0).to_frame()
    correlations.plot(kind='bar');


# In[60]:


feature_cor('total_cases')


# In[61]:


# Correlation matrix
corr = df11.corr()
plt.figure(num=None, figsize=(10, 10), dpi=200, facecolor='w', edgecolor='k')
corrMat = plt.matshow(corr, fignum = 1)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.gca().xaxis.tick_bottom()
plt.colorbar(corrMat)
plt.title(f'Correlation Matrix for the dataset', fontsize=15)
plt.show()


# Plot the scatter and density for all features

# In[62]:


df_temp= df11.select_dtypes(include =[np.number]) # keep only numerical columns
ax = pd.plotting.scatter_matrix(df_temp, alpha=0.75, figsize=[20, 20], diagonal='kde')
corrs = df_temp.corr().values
for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
    ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=10)
plt.suptitle('Scatter and Density Plot')
plt.show()


# Plot clustermap from the similarity matrix of the impact of non-pharmaceutical interventions on one of Total or New feature in dataset among 10 countries have top rank in the selected feature

# In[63]:


from tqdm import tqdm
from dtaidistance import dtw
def similarity_matrix(df,non_pharmaceutical_factor,total_new):
    selected_countries = df.groupby('iso_code')[total_new].max().sort_values(ascending=False).iloc[:10].index
    df_temp=df[['iso_code','date',non_pharmaceutical_factor]].set_index('date')
    factor_train = {}

    for country in selected_countries : 
            factor = df_temp[df_temp['iso_code']==country].copy().sort_index()
            factor_train[country]=factor

    factor_train_selection = []
    for my_country in factor_train.keys():
        if (factor_train[my_country][non_pharmaceutical_factor].std()>0):
            factor_train_selection.append(my_country)

    n=len(factor_train_selection)
    similarity_matrix = np.empty([n,n])
    for i,first_country in tqdm(enumerate(factor_train_selection)):
        for j,second_country in enumerate(factor_train_selection):
            if j>=i:
                first_factor = factor_train[first_country][non_pharmaceutical_factor]
                second_factor = factor_train[second_country][non_pharmaceutical_factor]
                distance = dtw.distance(first_factor,second_factor)
                similarity_matrix[i][j],similarity_matrix[j][i] = distance, distance

    df_similarity_matrix = pd.DataFrame(similarity_matrix, columns=factor_train_selection, index=factor_train_selection)
#     fig, ax = plt.subplots(figsize=(10,10)) 
    sns.clustermap(df_similarity_matrix)


# In[64]:


similarity_matrix(df11,'stringency_index','total_cases')


# ## Model Building Phase
# 1. Prophet library.
# 2. SARIMA.
# 3. SARIMAX.

# ### 1- Prophet Prediction

# We use Prophet, a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well. It is also an open source software released by Facebook’s Core Data Science team. It is available for download on CRAN and PyPI.
# 
# Why Prophet?
# 
# Prophet is easy to customize and use, and to produce accurate forecasts which can be explained intuitively with supporting evidence such as forecast seasonality components. It allows the analyst to explain in an intuitive and convinving manner to higher management as to why the forecasts are as such, and the plausible underlying factors that contribute to its result. Furthermore, it is also open-source!
# 
# References
# 
# https://facebook.github.io/prophet/
# 
# https://facebook.github.io/prophet/docs/
# 
# https://github.com/facebook/prophet

# #### Prophet with Daily & Weekly Seasonality (with custom Fourier orders):
# 
# Prophet will by default fit weekly and yearly seasonalities, if the time series is more than two cycles long. It will also fit daily seasonality for a sub-daily time series. You can add other seasonalities (monthly, quarterly, hourly) using the add_seasonality method (Python) or function (R).
# 
# The inputs to this function are a name, the period of the seasonality in days, and the Fourier order for the seasonality. For reference, by default Prophet uses a Fourier order of 3 for weekly seasonality and 10 for yearly seasonality. An optional input to add_seasonality is the prior scale for that seasonal component - this is discussed below.
# 
# Source: https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#fourier-order-for-seasonalities

# In[65]:


get_ipython().system(' pip install Prophet')


# In[66]:


from prophet import Prophet


# As we are now forecasting at country level, for small values, it is possible for forecasts to become negative. To counter this, we round negative values to zero. To perform forecast evaluations using mean absolute error (MAE), we require to partition the dataset into train & validation sets. Here, the test set will contain the dates for which the Prophet model is trained on and where forecasts were made.

# In[67]:


def prophet_prediction(df,country,days_to_forecast=7,first_forecasted_date='2022-01-03',mode = 'default',plot=1):
    features_to_predict=['new_cases','new_deaths','new_tests','new_vaccinations']
    mean_absolute_errors=[]
    first_forecasted_date=pd.to_datetime(first_forecasted_date)
    df_loc=df[df.location==country]
    df_loc['date']=pd.to_datetime(df_loc['date'])

#     forecast_dfs has actual and predicted values and its MAE
    forecast_dfs=df_loc[df_loc['date']>=first_forecasted_date][['location','date']].rename(columns={'date': 'ds'})
    for feature in features_to_predict:
        df_temp=df_loc[['date',feature]].rename(columns={feature: 'y', 'date': 'ds'})

#        Prepare the train and test datasets
        train = df_temp[df_temp.ds <pd.to_datetime(first_forecasted_date)]
        test = df_temp[df_temp.ds >=pd.to_datetime(first_forecasted_date)]

#         Forecasting the feature with Prophet by the country
#         With baseline Prophet Model, using default parameters
        if mode == 'default':
            model = Prophet()
#         With custom seasonalities & Fourier orders
        elif mode == 'custom':
            model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
            model.add_seasonality(name='monthly', period=30.5, fourier_order=10)
            model.add_seasonality(name='weekly', period=7, fourier_order=21)
            model.add_seasonality(name='daily', period=1, fourier_order=3)
        model.fit(train)
        future = model.make_future_dataframe(periods=days_to_forecast)
        forecast = model.predict(future)
    
#         save the forcasted values and its MAE
        forecast_df = forecast[['ds', 'yhat']]
        result_df = forecast_df[(pd.to_datetime(forecast_df['ds']) >= first_forecasted_date)]
        result_val_df = result_df.merge(test, on=['ds'])
        result_val_df['absolute_errors'] = (result_val_df['y'] - result_val_df['yhat']).abs()
        mean_absolute_errors += list(result_val_df['absolute_errors'].values)
#         Round negative values to zero
        result_val_df.yhat=result_val_df.yhat.mask(result_val_df.yhat.lt(0),0)
#         Round decimal values to int
        result_val_df.yhat=result_val_df.yhat.round()
        result_val_df=result_val_df.rename(columns={'y':feature, 'absolute_errors':f'{feature}_absolute_errors','yhat':f'{feature}_predicted'})
        forecast_dfs=pd.merge(forecast_dfs,result_val_df,on=('ds'))

#         Plot actual vs. predicted results for the feature in the country
        if plot==1:
            fig = go.Figure(
                layout=go.Layout(title=go.layout.Title(
                    text=f"Actual vs. Predicted results for {feature} in {country}")))
            fig.add_trace( go.Scatter(x=forecast.ds, y=forecast.yhat, error_y=dict(
                        type='data', # value of error bar given in data coordinates
                        symmetric=False,
                        array=forecast.yhat_upper - forecast.yhat,
                        arrayminus=forecast.yhat - forecast.yhat_lower,
                        visible=True), name='Forecast'))
            fig.add_trace( go.Scatter( x=train['ds'].values, y=train['y'].values, name='Actual'))
            fig.show()

#      Change the date column name   
    forecast_dfs=forecast_dfs.rename(columns={'ds':'date'})
#     Calculate the mean of all mean_absolute_errors
    mean_absolute_errors=np.mean(mean_absolute_errors)
            
    return forecast_dfs,mean_absolute_errors


# In[68]:


forecast_dfs_default,mae_default=prophet_prediction(df11,'Canada',days_to_forecast=7,first_forecasted_date='2022-01-03',mode = 'default',plot=1)


# In[69]:


forecast_dfs_custom,mae_custom=prophet_prediction(df11,'Canada',days_to_forecast=7,first_forecasted_date='2022-01-03',mode = 'custom',plot=1)


# In[70]:


forecast_dfs_custom


# In[71]:


forecast_dfs_default


# In[72]:


mae_custom


# In[73]:


mae_default


# ### 2- SARIMA Model

# In[74]:


# df=df.drop(['Unnamed: 0'],axis=1)
df=df11[df11['location']=="Canada"]
df=df[['date','new_cases']].set_index('date')


# Time series Stationary check

# In[75]:


from statsmodels.tsa.stattools import adfuller


# In[76]:


def ad_test(df):
    dftest=adfuller(df,autolag='AIC')
#     print adfuller test P-value
    print(dftest[1])


# In[77]:


ad_test(df['new_cases'])


# The new cases time series is not stationary. To be stationary, P-value should be less than 0.05

# In[78]:


from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')


# Use auto_arima to find best model

# In[79]:


best_model=auto_arima(df['new_cases'],trace=True,suppress_warnings=True)
best_model.summary()


# In[80]:


import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# evaluation
from sklearn import metrics
# plotly offline
# from plotly.offline import download_plotlyjs,init_notebook_mode
# init_notebook_mode(connected=True)


# Decompose the Time series: Trend, Seasonality, error/white noise

# In[81]:


plt.style.use('seaborn');
decompose = seasonal_decompose(df['new_cases'],model='additive',period=14);
decompose.plot();


# Plot ACF: auto-correlation function

# In[82]:


plot_acf(df['new_cases'].diff().dropna(),lags=50);


# Plot PACF: partial auto-correlation function

# In[83]:


plot_pacf(df['new_cases'].diff().dropna(),lags=50);


# SARIMA model with p=4, d=1, q=5 and seasonal_order=7

# In[84]:


def SARIMA_model(df,country,test_size,p,d,q,s,endog='new_cases', plot=1):
    df_temp=df[df['location']==country]
    df_temp=df_temp[['date',endog]].set_index('date')
    train=df_temp.iloc[:-test_size]
    test=df_temp.iloc[-test_size:]
    SARIMA_model = sm.tsa.statespace.SARIMAX(train.values,
                                                order=(p, d, q),
                                                seasonal_order=(0,0,0,s),
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
    SARIMA_model = SARIMA_model.fit(maxiter=1000)
#     Forecast
    SARIMA_forecast =SARIMA_model.get_forecast(steps=test_size)
    SARIMA_pred = SARIMA_forecast.summary_frame(alpha=0.05).set_index(pd.date_range(start=test.index[0], periods=test_size, freq='D'))
#     Evaluate the model by RMSE
    SARIMA_RSME = np.sqrt(metrics.mean_squared_error(test,SARIMA_pred['mean']))
    
    if plot==1:
        df_temp=df_temp.tail(30)
        fig = go.Figure(
        layout=go.Layout(title=go.layout.Title(
            text=f"Actual vs. Predicted results for {endog} in {country}")))
        fig.add_trace(go.Scatter(name="Actual",
             x=df_temp.index, y=df_temp[endog]))
        fig.add_trace(go.Scatter(name="Predicted",
             x=SARIMA_pred.index, y=SARIMA_pred['mean']))
        fig.add_trace(go.Scatter(name="Lowerbound",
            line=dict(width=0),fill='tozeroy', opacity=0.1, showlegend=False,
            x=SARIMA_pred.index, y=SARIMA_pred['mean_ci_lower']))
        fig.add_trace(go.Scatter(name="Upperbound",
            fill='tozeroy', opacity=0.1, showlegend=False,
            x=SARIMA_pred.index, y=SARIMA_pred['mean_ci_upper']))
        fig.show()
        
    return SARIMA_model,SARIMA_pred,SARIMA_RSME


# In[85]:


SARIMA_model,SARIMA_pred,SARIMA_RSME=SARIMA_model(
    df11,"Canada",test_size=7,p=4,d=1,q=5,s=7,plot=1)


# In[86]:


# Model summary
SARIMA_model.summary()


# In[87]:


SARIMA_pred


# In[88]:


SARIMA_RSME


# In[89]:


plt.style.use('seaborn')
SARIMA_model.plot_diagnostics()


# ### 3- SARIMAX Model

# In[90]:


def SARIMAX_model(df,country,test_size,p,d,q,s,endog='new_cases',exog=['new_deaths','new_tests'],plot=1):
    df_temp=df[df['location']==country]
    df_temp_features_list=['date',endog]+exog
    df_temp=df_temp[df_temp_features_list].set_index('date')
    train=df_temp.iloc[:-test_size]
    test=df_temp.iloc[-test_size:]
    
    SARIMAX_model = sm.tsa.statespace.SARIMAX(train[endog].values,exons=train[exog],
                                            order=(p,d,q),
                                            seasonal_order=(p,d,q,s),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False,)

    SARIMAX_model = SARIMAX_model.fit(maxiter=1000)
#     Forecast
    SARIMAX_forecast =SARIMAX_model.get_forecast(steps=test_size)
    SARIMAX_pred = SARIMAX_forecast.summary_frame(alpha=0.05).set_index(pd.date_range(start=test.index[0], periods=test_size, freq='D'))

#     Plot the prediction results
    if plot==1:
        df_temp=df_temp.tail(30)
        fig = go.Figure(
        layout=go.Layout(title=go.layout.Title(
            text=f"Actual vs. Predicted results for {endog} in {country}")))
        fig.add_trace(go.Scatter(name="Actual",
             x=df_temp.index, y=df_temp[endog]))
        fig.add_trace(go.Scatter(name="Predicted",
             x=SARIMAX_pred.index, y=SARIMAX_pred['mean']))
        fig.add_trace(go.Scatter(name="Lowerbound",
            line=dict(width=0),fill='tozeroy', opacity=0.1, showlegend=False,
            x=SARIMAX_pred.index, y=SARIMAX_pred['mean_ci_lower']))
        fig.add_trace(go.Scatter(name="Upperbound",
            fill='tozeroy', opacity=0.1, showlegend=False,
            x=SARIMAX_pred.index, y=SARIMAX_pred['mean_ci_upper']))
        fig.show()
    return SARIMAX_model,SARIMAX_pred


# In[91]:


SARIMAX_model,SARIMAX_pred=SARIMAX_model(
    df11,"Canada",test_size=7,p=5,d=1,q=4,s=7,endog='new_cases',exog=['new_deaths','new_tests'],plot=1)


# In[92]:


# Model summary
SARIMAX_model.summary()


# In[93]:


SARIMAX_pred


# In[94]:


plt.style.use('seaborn')
SARIMA_model.plot_diagnostics()


# I created a pretty nice dashboard to visualize this data using tableau:
# [Dashboard](https://public.tableau.com/app/profile/husam.almasri/viz/Covid-19_16433519827770/Dashboard1)
