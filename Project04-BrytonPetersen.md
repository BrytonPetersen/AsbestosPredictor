# Asbestos risk prediction 

__Bryton Petersen__
__03/06/2021__




## Elevator pitch

The health related issues caused by asbestos were discovered originally by German scientists as far back as the late 1930's, but because of the poor political relationship between Germany and the United States at the time American scientists were reluctant to admit that they were right. By the 1960's it became apparent that serious diseases such as cancer were sometimes byproducts of exposure to asbestos - but the growing asbestos industry ignored and donwplayed the health risks caused by their product. By the 1980's asbestos regulations had downsized the industry and the hazards of the mineral was widespread. In an attempt to find homes that may still use asbestos as insulation, I used machine learning to help me sort between homes made pre and post 1980.


## Technical Details

__Grand Question 1__  -  Evaluating potential relationships between the home variables and the year the home was built.

__Features of Homes Built in Different Years__

![](GrandQuestion01.png)

We are able to see a couple of very clear relationships between home variables and the year the home was built in the chart above. In the first chart we can see that if a home was built with 0, 1, 2, or 5 bedrooms it almost always dates pre 1980. Homes that date post 1980 are much more likely to have 3 or 4 bedrooms. 

The second chart shows us that homes built pre 1980 were most likely to have 1 or 2 bathrooms. Many of them also had 3 baths and 6 baths, but it is more likely homes built post 1980 have 3 or 6 baths. If a house has 4 or 5 baths they were almost always built post 1980's (in this dataset).

The third chart compares the number of stories houses were built with. The data portrayed in this chart tells us that homes built with only 1 story were often made between 1930 and 1950 but from the 1960's to the 2000's multi story homes were much more common. In the late 1990's and early 2000's single story homes began to make a re-apperance 


__Grand Question 2__ - Finding the most accurate classification model.

After trial and error of 7 different classification models, including bagging, random forest, extra randomized trees, Adaboost, gradient tree boosting, histogram-based gradient boosting, and a voting classifier, I found the random forest classification model to return the best results with my given dataset. The prediction yeilded with the random forest classification had an accuracy of 92%. 


__Grand Question 3__ - Justifying my classification model

CHART 1 - Feature Importance

![](important.png)

When viewing relationships (see CHART 1) between features of homes and the year that homes were built in we saw that two of the most dramatic differences between homes built pre and post 1980 were the number of stories they had and the number of baths they had. If we look at the chart above we can see that some of the most important features used to help differentiate between homes built pre and post 1980 in our machine learning model were the number of stories and the number of baths in a home. This lets me know that the classification model being used (RandomForestClassification) is at least on the right track. 

__Grand Question 4__ - Proving the accuracy of the RandomForest classifier

              precision    recall  f1-score   support

           0       0.90      0.89      0.90      2553
           1       0.94      0.94      0.94      4321

    accuracy                           0.92      6874


 Using a classification report (code shown below) I was able to produce the accuracy of the predictions made by the RandomForestClassifier. This method compares the predictions made by my model with the actual data (namely, what year the house was built in). 0 in this classification report represents the houses that my classifier predicted as built before 1980, and 1 represents the homes built after 1980. It was 90% precise in its predictions for homes built pre 1980 and correctly predicted 2553 homes, and 94% precise in its predictions for homes built post 1980 where it correctly predicted 4321 homes out of the dataset.

 As for accuracy, this classifier had an accuracy rate of 92% and correctly predicted 6841 homes out of the dataset. 

 ```python
 print(metrics.classification_report(y_test, predictions))
 ```
 
 ## APPENDIX A (PYTHON SCRIPT)

```python
# %%

#These are my imports from sklearn

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# %%

#These are my basic imports

import pandas as pd 
import altair as alt
import numpy as np
import seaborn as sns

# %%

#Get my house data

dwellings_denver = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_denver/dwellings_denver.csv")
dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
dwellings_neighborhoods_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")


#Fix the 5000 rows error with altair
alt.data_transformers.enable('json')

# %%

#Create a test sample and a train sample from my database

X_pred = dwellings_ml.drop(columns = ['yrbuilt','before1980'])
y_pred = dwellings_ml.filter(['before1980'])
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, 
    y_pred, 
    test_size = .30, 
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
# Specify which classifier is going to be used and 'train' the data using the fit funcion

den_c = ensemble.RandomForestClassifier()
den_c_1= den_c.fit(X_train, y_train)

# %%

# Use the classifier to make predictions given the test data

predictions = den_c.predict(X_test)
```