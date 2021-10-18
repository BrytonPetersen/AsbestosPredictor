# %%
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

"""
These are my basic imports
"""
import pandas as pd 
import altair as alt
import numpy as np
import seaborn as sns

# %%
dwellings_denver = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_denver/dwellings_denver.csv")
dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
dwellings_neighborhoods_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")
alt.data_transformers.enable('json')
# %%
subset = dwellings_ml.filter(['livearea', 'finbsmnt', 
    'basement', 'yearbuilt', 'nocars', 'numbdrm', 'numbaths', 
    'stories', 'yrbuilt', 'before1980']).sample(500)
sns.pairplot(subset, hue = 'before1980')
corr = subset.drop(columns = 'before1980').corr()

# %%
X_pred = dwellings_ml.drop(columns = ['yrbuilt', 'before1980'])
y_pred = dwellings_ml.filter(['before1980'])
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, 
    y_pred, 
    test_size = .34, 
    random_state = 76)

# %%
"""
Question 1
"""
bathroom = (alt.Chart(subset)
.encode(
    alt.X('numbaths:O'),
    alt.Y('yrbuilt',
    scale = alt.Scale(zero=False),
    axis = alt.Axis(format = 'd'),
    title = 'Year Built'))
.mark_boxplot()
.properties(title = 'Bathroom vs. Year'))

bedroom = (alt.Chart(subset)
.encode(
    alt.X('numbdrm:O', title = "Number of Bedrooms"),
    alt.Y('yrbuilt',
    scale = alt.Scale(zero=False),
    axis = alt.Axis(format = 'd'),
    title = 'Year Built'))
.mark_boxplot()
.properties(title = 'Bedroom vs. Year'))

stories = (alt.Chart(dwellings_ml)
.encode(
    alt.X('arcstyle_ONE-STORY:O', title = "One story or not"),
    alt.Y('yrbuilt',
    scale = alt.Scale(zero=False),
    axis = alt.Axis(format = 'd'),
    title = 'Year Built'))
.mark_boxplot()
.properties(title = 'Stories'))

q1 = alt.hconcat(bedroom, bathroom, stories, title = 'Grand Question 1')
# %%
