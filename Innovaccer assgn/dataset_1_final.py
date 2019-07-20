
# coding: utf-8

# # TripAdvisor Dataset

# In[222]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# # Reading the dataset and basic processing

# In[223]:


data = pd.read_excel("./Potential datasets for recruitment.xlsx", sheet_name=0)
data.head()


# In[224]:


data.dtypes


# In[225]:


data.count()


# No NaN values, data seems to be pretty solid here,except for one stray *Member years* value:

# In[226]:


data[data['Member years'] == -1806]


# However, we leave it as is right now, until it proves to be an important factor.

# In[227]:


data.Score.mean() #Data has lots of high ratings. Mildly skewed.


# In[228]:


bin_dict = {'YES' : 1,
            'NO' : 0}
cols_to_replace = ['Swimming Pool', 'Exercise Room', 'Basketball Court', 'Yoga Classes', 'Club', 'Free Wifi']
data.replace({x : bin_dict for x in cols_to_replace}, inplace=True)


# # Mining and Visualisation

# In[229]:


sns.heatmap(data.corr())


# In[10]:


# data['User country'].value_counts()
# data['Nr. reviews'].median()
# data.Score.unique()
# data['Period of stay'].unique()
# data['Traveler type'].unique()
# data['Hotel stars'].unique()


# In[11]:


for x in data.columns:
    plt.figure()
    sns.countplot(data=data, y=x, hue='Score')


# Let's consolidate some facts here. 
# * Hotels in this dataset are generally rated towards the higher side. The mean, median are around 4 stars.
# * No of hotels are equally distributed in this dataset (pretty convenient!).
# * The *traveler type* could be a viable important measure.
# * In the amenities, *Free Wifi* and *Swimming Pool* seem to be important factors for getting a 5 star rating.
# * 5 and (4,5),(3,5) star hotels have more 5 star scores.
# * The *User continent* feature should be examined closely.

# In[250]:


cols_to_examine =  ['Traveler type','Swimming Pool',
                   'Exercise Room','Yoga Classes','Basketball Court', 'Club', 'Free Wifi',
                   'Hotel stars', 'User continent']


# In[242]:


g = data.groupby('Helpful votes').agg(['count', 'mean', 'median'])['Score']
plt.figure()
sns.scatterplot(x=g.index, y=g['mean'])
plt.title('Helpful votes vs Mean of scores')


# In[246]:


sns.boxplot(x=data['Period of stay'], y=data['Score'])


# In[247]:


sns.boxplot(x=data['Traveler type'], y=data['Score'])


# In[248]:


sns.boxplot(x=data['Hotel stars'], y=data['Score'])


# In[249]:


sns.boxplot(x=data['User continent'], y=data['Score'])


# The values are generally inclined to the higher side, and 1 and 2 ratings are barely outliers in some cases. This points to a heavily skewed dataset.

# In[251]:


for x in cols_to_examine:
    g = data.groupby(x).agg(['count', 'mean', 'median'])['Score']
    print(g)


# The above data simply verifies our previous assumptions, that *Traveler type, Swimming Pool, Basketball Court*(to some extent)*, Free Wifi, Hotel stars*, and (maybe) *User Continent* have a major part to play in deciding the Score for the hotel.

# In[14]:


data.columns


# In[15]:


insig_cols = ['Nr. reviews', 'Nr. hotel reviews', 'Hotel name','Nr. rooms','Member years', 'Review month', 'Review weekday']
for x in insig_cols:
    g = data.groupby(x).agg(['count', 'mean', 'median'])['Score']
    print(g)


# The *Hotel name* is the only feature that stands out a little bit, but that could be because of the various factors that go into making the hotel what it is.

# We could also classify the hotel based on whether it contains a casino or not.

# In[253]:


data['Has casino'] = data['Hotel name'].str.contains('^.*Casino.*$')


# In[254]:


data['Has casino'].replace({True:1, False:0}, inplace=True)


# In[255]:


sns.countplot(data=data, y = 'Has casino', hue = 'Score')
plt.title("Count of Scores wrt Casino presence")


# In[28]:


g = data.groupby('Has casino').agg(['count', 'mean', 'median'])['Score']
g


# It does seem to make a difference. However, the count difference is also significant.

# #### Some more columns

# Let's make columns for total amenities of the hotel, along with some combinations of important amenities.

# In[33]:


data['Amenities count'] = data['Swimming Pool'] + data['Exercise Room'] + data['Basketball Court'] + data['Yoga Classes']+ data['Club'] + data['Free Wifi'] + data['Has casino']


# In[34]:


sns.countplot(data=data, y='Amenities count', hue='Score')


# In[36]:


g = data.groupby('Amenities count').agg(['count', 'mean', 'median'])['Score']
g


# No. of amenities don't seem to make much of a difference.

# In[38]:


g = data.groupby([ 'Hotel name','Period of stay']).agg(['count', 'mean', 'median'])['Score']
g


# This conjunction is also not very significant.

# In[39]:


amen_cols = ['Swimming Pool',
       'Exercise Room', 'Basketball Court', 'Yoga Classes', 'Club',
       'Free Wifi','Has casino']


# In[43]:


data1 = data.copy()
for x in amen_cols:
    for y in amen_cols:
        if x != y:
            col_name = x+'&'+y
            data1[col_name] = data1[x] & data1[y]
            plt.figure()
            sns.countplot(data=data1, y=col_name, hue='Score')
            g = data1.groupby(col_name).agg(['count', 'mean', 'median'])['Score']
            print(g)


# The usual suspects, no very noteworthy inclusions in the conjunctions.

# # Fit and train the model

# In[192]:


data.columns


# In[193]:


MODEL_COLUMNS = [ 'Period of stay', 'Traveler type', 'Swimming Pool',
       'Exercise Room', 'Basketball Court', 'Yoga Classes', 'Club',
       'Free Wifi', 'Hotel name', 'Hotel stars', 'User continent','Has casino']


# In[194]:


# data1 = data.copy()
data1 = pd.get_dummies(data, columns=['User country','Period of stay','Traveler type','Hotel name', 'Hotel stars',
                                      'User continent', 'Review month', 'Review weekday'])


# In[195]:


data1.columns


# In[200]:


data1.drop(index=75, inplace=True)


# In[201]:


data1.columns


# In[208]:


from sklearn.feature_selection import SelectKBest, chi2, f_classif
data1.columns[SelectKBest(f_classif, k=11).fit(data1, data1['Score']).get_support()]


# In[107]:


MODEL_COLS_1 = ['Swimming Pool', 'Exercise Room', 'Basketball Court',
       'Yoga Classes', 'Club', 'Free Wifi','Has casino',
       'Traveler type_Business', 'Traveler type_Couples',
       'Traveler type_Families', 'Traveler type_Friends', 'Traveler type_Solo',
       'Hotel stars_3', 'Hotel stars_4', 'Hotel stars_5', 'Hotel stars_3,5',
       'Hotel stars_4,5']


# *MODEL_COLS_2* refers to the variables chosen by SelectKBest, using chi2 as the testing function.

# In[203]:


MODEL_COLS_2 = ['Nr. reviews', 'Nr. hotel reviews', 'Helpful votes',
       'Nr. rooms', 'Has casino', 'User country_China',
       'Hotel name_Circus Circus Hotel & Casino Las Vegas',
       'Hotel name_Monte Carlo Resort&Casino', 'Hotel stars_3',
       'Hotel stars_5']


# *MODEL_COLS_3* refers to the variables chosen by SelectKBest, using f_classif as the testing function.

# In[210]:


MODEL_COLS_3 = ['Swimming Pool', 'Free Wifi', 'Has casino',
       'User country_China', 'User country_Swiss',
       'Hotel name_Circus Circus Hotel & Casino Las Vegas',
       'Hotel name_Monte Carlo Resort&Casino', 'Hotel stars_3',
       'Hotel stars_4', 'Hotel stars_5']


# In[211]:


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
import mord as m
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# In[212]:


kf = KFold(n_splits=5, shuffle=True)


# In[213]:


clf = m.LogisticIT()


# In[214]:


indices = list(kf.split(data1[MODEL_COLS_3], data1['Score']))
count = 1
score = 0
for train_index, test_index in indices:
    train_index = list(train_index)
    test_index = list(test_index)
    X_train = data1[MODEL_COLS_3].iloc[train_index]
    y_train = data1.Score.iloc[train_index]
    X_test = data1[MODEL_COLS_3].iloc[test_index]
    y_test = data1.Score.iloc[test_index]
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    #print(predict[:10])
    predict = np.round(predict)
    #print(*zip(predict, y_test))
    print("Iter {0}: {1}".format(count, accuracy_score(predict, y_test)))
    score += mean_squared_error(predict, y_test)
    count += 1


# In[116]:


results_4 = []


# In[117]:


results_4.append(('mean_squared_error',score/5))


# In[118]:


results_4


# In[120]:


results_3


# The results are abysmal, whether we use SelectKBest(2 and 3) or our previous visualisations(1). They are as good as guessing the rating. Lets have a closer look at the predictions.

# In[122]:


print(*zip(predict, y_test))


# Our model doesn't even predict 1s and 2s.
# Use Stratified K-fold.

# In[140]:


kf = StratifiedKFold(n_splits=5, shuffle=True)


# In[141]:


clf = RandomForestRegressor()


# In[142]:


indices = list(kf.split(data1[MODEL_COLS_1], data1['Score']))
count = 1
score = 0
for train_index, test_index in indices:
    train_index = list(train_index)
    test_index = list(test_index)
    X_train = data1[MODEL_COLS_1].iloc[train_index]
    y_train = data1.Score.iloc[train_index]
    #print(y_train.mean())
    X_test = data1[MODEL_COLS_1].iloc[test_index]
    y_test = data1.Score.iloc[test_index]
    #print(y_test.mean())
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    #print(predict[:10])
    predict = np.round(predict)
    #print(*zip(predict, y_test))
    print("Iter {0}: {1}".format(count, mean_squared_error(predict, y_test)))
    score += mean_squared_error(predict, y_test)
    count += 1


# In[143]:


print(*zip(predict, y_test))


# The data is simply too skewed for any kind of stratification to work.

# So, the best result we found was by using Ordinal Regression on the data (~50% accuracy). The feature importances can be derived using RandomForestRegressor as:

# In[145]:


print(*zip(MODEL_COLS_1,clf.feature_importances_))


# There is no clear winner as to the feature importances, except the above made conclusions:
# * Amenities such as *Swimming Pool, Basketball Court, Free Wifi*, and *Has Casino*
# * Your *Traveler type*
# * If you are 4 or 5 star

# # Further analysis

# Let's check the differences between a 1 star and a 5 star rated hotel.

# In[152]:


data[data['Score'] == 1].sample(10)


# In[157]:


data[data['Score'] == 5].sample(10)


# With the naked eye, they look pretty similar.

# #### Hypothesis testing

# In[153]:


from scipy.stats import ttest_ind


# In[155]:


ttest_ind(data[data['Free Wifi'] == 1].Score, data[data['Free Wifi'] == 0].Score) #Proves that free wifi is a differentiator.


# In[156]:


from scipy.stats import f_oneway


# In[159]:


free_wifi = []
for x in range(1,6):
    free_wifi.append(list(data[data['Score'] == x]['Free Wifi']))


# In[161]:


f_oneway(*free_wifi)


# In[162]:


#Proves that different scores have different quantities of free wifi values.


# Since this is significant too, our lack of a good result can only be attributed to a lack of data, as well as heavily skewed data, which could not be corrected even by stratification. We didn't try to over/under sample due to the lack of data. The important features have been summarised above already. 
# Also, the low scored and high scored hotels have similar aggregated features, so the stars could also be a result of simply the user experience, which cannot be covered by these variables. 

# # Conclusion

# We were unable to derive a convincing result out of the predictions using the data present due to a variety of factors:
# * Heavily left skewed dataset(Higher values were generally preferred)
# * Small data size
# * Very less significant continuous features
# * Ratings are generally affected by user experience, which was absent from the feature list

# Analytically, the most important features turned out to be:
# 
# * Amenities such as *Swimming Pool, Basketball Court, Free Wifi*, and *Has Casino*
# * Your *Traveler type*
# * If you are 4 or 5 star
