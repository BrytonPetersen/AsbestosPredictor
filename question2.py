# %%
"""
These are my imports from sklearn
"""
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# %%
"""
These are my basic imports
"""
import pandas as pd 
import altair as alt
import numpy as np
import seaborn as sns

# %%
"""
Get my house data
"""
dwellings_denver = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_denver/dwellings_denver.csv")
dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
dwellings_neighborhoods_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")
"""
Fix the 5000 rows error with altair
"""
alt.data_transformers.enable('json')

# %%
#Make a subset
h_subset = dwellings_ml.filter(['livearea', 'finbsmnt', 
    'basement', 'yearbuilt', 'nocars', 'numbdrm', 'numbaths', 
    'stories', 'yrbuilt', 'before1980'])



# %%
#DONT RUN THIS AGAIN
'''sns.pairplot(dwellings_ml, hue = 'before1980')
corr = dwellings_ml.drop(columns = 'before1980').corr()'''

# %%
X_pred = dwellings_ml.drop(columns = ['yrbuilt','before1980'])
y_pred = dwellings_ml.filter(['before1980'])
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, 
    y_pred, 
    test_size = .30, 
    random_state = 76) 

# %%
den_c = ensemble.RandomForestClassifier()
den_c_1= den_c.fit(X_train, y_train)
# %%
predictions = den_c.predict(X_test)

# %%
score = den_c.score(X_test, y_test)
print(score)
metrics.accuracy_score(y_test, predictions)
# %%
metrics.recall_score(y_test, predictions)
# %%
metrics.precision_score(y_test, predictions)
# %%
print(metrics.classification_report(y_test, predictions))
# %%



# %%
accuracy = metrics.plot_roc_curve(den_c, X_test, y_test)
# %%
feature_df = pd.DataFrame({'features':X_train.columns,
 'importance':den_c.feature_importances_}).sort_values(by = 'importance', ascending = False).head(25)

# %%
values = ['arcstyle_ONE-STORY','livearea','stories','numbaths','tasp','netprice','sprice']
#good way to print out the names of each column in feature_df
#print(*feature_df.features, sep = ', ')
# %%
chart_3 = (alt.Chart(feature_df).mark_bar().encode(
    y = alt.Y('features', sort = values),
    x = alt.X('importance')
)
.properties( title = 'Most Important Features'))

chart_3

# %%
