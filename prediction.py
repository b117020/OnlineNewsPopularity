# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:16:05 2020

@author: Devdarshan
"""

'''
Dataset Description:
n_tokens_title: Number of words in the title
n_tokens_content: Number of words in the content
n_unique_tokens: Rate of unique words in the content
n_non_stop_unique_tokens: Rate of unique non-stop words in the content
num_hrefs: Number of links
num_imgs: Number of images
num_videos: Number of videos
average_token_length: Average length of the words in the content
num_keywords: Number of keywords in the metadata
data_channel_is_lifestyle: Is data channel 'Lifestyle'?
data_channel_is_entertainment: Is data channel 'Entertainment'?
data_channel_is_bus: Is data channel 'Business'?
data_channel_is_socmed: Is data channel 'Social Media'?
data_channel_is_tech: Is data channel 'Tech'?
data_channel_is_world: Is data channel 'World'?
weekday_is_monday: Was the article published on a Monday?
weekday_is_tuesday: Was the article published on a Tuesday?
weekday_is_wednesday: Was the article published on a Wednesday?
weekday_is_thursday: Was the article published on a Thursday?
weekday_is_friday: Was the article published on a Friday?
weekday_is_saturday: Was the article published on a Saturday?
weekday_is_sunday: Was the article published on a Sunday?
global_subjectivity: Text subjectivity
global_sentiment_polarity: Text sentiment polarity
global_rate_positive_words: Rate of positive words in the content
global_rate_negative_words: Rate of negative words in the content
avg_positive_polarity: Avg. polarity of positive words
min_positive_polarity: Min. polarity of positive words
max_positive_polarity: Max. polarity of positive words
avg_negative_polarity: Avg. polarity of negative words
min_negative_polarity: Min. polarity of negative words
max_negative_polarity: Max. polarity of negative words
title_subjectivity: Title subjectivity
title_sentiment_polarity: Title polarity
shares: Number of shares (target)
'''
#imports
import pandas as pd
#from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model


#data cleaning
data = pd.read_csv('OnlineNewsPopularity.csv').rename(columns=lambda x: x.strip())
data=data.drop(['url','timedelta', 'LDA_00','LDA_01','LDA_02','LDA_03','LDA_04','num_self_hrefs', 'kw_min_min', 'kw_max_min', 'kw_avg_min','kw_min_max','kw_max_max','kw_avg_max','abs_title_subjectivity','kw_min_avg','kw_max_avg','kw_avg_avg','n_non_stop_words','self_reference_min_shares','abs_title_sentiment_polarity','self_reference_max_shares','rate_positive_words','rate_negative_words','self_reference_avg_sharess','is_weekend'], axis=1)

#Scaling/standardizing data
def scaler(data):
    scaler = MinMaxScaler()
    X=data.drop('shares', axis=1)
    X[X.columns] = scaler.fit_transform(X[X.columns])
    y=data['shares']
    return X,y

X,y=scaler(data)
#train test split
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, random_state=123)

'''
#BayesianRidge
bay_rid = BayesianRidge()
bay_rid.fit(train_X, train_y)
bay_predictions = bay_rid.predict(test_X)
print("Root mean squared error: %f" % mean_squared_error(test_y, bay_predictions))
print("Mean absolute error: %f" % mean_absolute_error(test_y, bay_predictions))

#lasso
lasso = Lasso()
lasso.fit(train_X, train_y)
lasso_predictions = lasso.predict(test_X)
print("Root mean squared error: %f" % mean_squared_error(test_y, lasso_predictions))
print("Mean absolute error: %f" % mean_absolute_error(test_y, lasso_predictions))

#ridge
ridge = Ridge(alpha=0.5)
ridge.fit(train_X, train_y)
ridge_predictions = ridge.predict(test_X)
print("Root mean squared error: %f" % mean_squared_error(test_y, ridge_predictions))
print("Mean absolute error: %f" % mean_absolute_error(test_y, ridge_predictions))

#LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_X, train_y)
linear_predictions = lin_reg.predict(test_X)
print("Root mean squared error: %f" % mean_squared_error(test_y, linear_predictions))
print("Mean absolute error: %f" % mean_absolute_error(test_y, linear_predictions))
'''

#deep learning model 

n_cols = X.shape[1]
model = Sequential()

# The Input Layer 
model.add(Dense(32, kernel_initializer='normal',input_dim = n_cols, activation='relu'))

# The Hidden Layers 
model.add(Dense(32, kernel_initializer= 'normal',activation='relu'))
model.add(Dense(64, kernel_initializer='normal',activation='relu'))

# The Output Layer 
model.add(Dense(1,activation='linear'))

# Compile the network 
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

#model_fitting
model.fit(train_X, train_y, epochs=100)
#model.save("Virality_prediction.h5")
model = load_model("virality_prediction.h5")

predictions = model.predict(test_X)

#metrics
print("Root mean squared error: %f" % sqrt(mean_squared_error(test_y, predictions))) #14719
print("Mean absolute error: %f" % mean_absolute_error(test_y, predictions)) #2479

