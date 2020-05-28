# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:35:56 2020

@author: Devdarshan
"""
#imports
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from statistics import mean 
#import urllib.request
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from datetime import date,datetime
import calendar


#create lists to store url,title and date of publication of blogs
clink = []
ctitle = []
cdate = []
blog_content = []
ctags=[]

#features list
all_links_count=[]
day=[]
title_words_count=[]
contents_words_count=[]
unique_words_count=[]
unique_non_stop_words=[]
avg_length_words = []
num_keywords=[]
subjectivity=[]
polarity=[]
pos_words=[]
neg_words=[]
avg_pos_pol=[]
avg_neg_pol=[]
max_pos_pol=[]
max_neg_pol=[]
min_pos_pol=[]
min_neg_pol=[]
image_count = [0] * 10
video_count = [0]*10
title_sub =[]
title_pol =[]
video_count = [0]*10
self_hrefs = [0]*10
data_channel_is_lifestyle= [0]*10
data_channel_is_entertainment= [0]*10
data_channel_is_bus= [0]*10
data_channel_is_socmed =[0]*10
data_channel_is_tech =  [1]*10
data_channel_is_world =[0]*10

#load driver to use chrome and scrape required contents from careeranna(Infromational website)
browser = webdriver.Chrome("D:\downloads\Downloads//chromedriver_win32 (1)//chromedriver.exe")
for i in range(0,1):
    print(i)
    browser.get("https://www.careeranna.com/articles/category/mba/pages/"+str(i))
    elem = browser.find_element_by_tag_name("body")
    title = browser.find_elements_by_class_name("pt-cv-title")
    link = browser.find_elements_by_class_name("pt-cv-href-thumbnail")
    tags = browser.find_elements_by_class_name("terms")
    date =  browser.find_elements_by_tag_name("time")
    for post in title:
        if len(post.text)==0 :
            continue
        else :
            ctitle.append(post.text)
    for post in link:
        clink.append(post.get_attribute('href'))
        
    for post in date:
        cdate.append(post.text)
for i in range(10):
    cdate[i]=cdate[i].replace(',', '')
    #cdate[i]=datetime.datetime.strptime(cdate[i], '%d/%m/%Y')
    day.append(calendar.day_name[datetime.strptime(cdate[i], '%d/%m/%Y').weekday()])

for i in range(1):
    print(i)
    browser.get("https://www.careeranna.com/articles/category/mba/pages/"+str(i))
    elem = browser.find_element_by_tag_name("body")
    tags = browser.find_elements_by_class_name("terms")
    for post in tags:
        if len(post.text)==0 :
            ctags.append(' ')
        else :
            ctags.append(post.text)

for i in range(10):
    num_keywords.append(len(ctags[i].split()))
    
links = clink
#scrape blog content 
for i in range(0,10):
    print(i)
    url = links[i]
    res = requests.get(url)
    html_page = res.content


    soup = BeautifulSoup(html_page, 'html.parser')

    text = soup.find_all(text=True)
    set([t.parent.name for t in text])

    output = ''
    list1 = [
	'a',
 'article',
 'b',
 'body',
 'cite',
 'div',
 'em',
 'footer',
 'form',
 'h1',
 'h2',
 'h3',
 'h6',
 'head',
 'header',
 'i',
 'p',
 'span',
 'strong',
 
    ]

    for t in text:
        if t.parent.name in list1:
            print(t)
            output += '{} '.format(t)
    blog_content.append(output)
    
#cleaning of scraped content
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)
for i in range(0,10):
    blog_content[i]=remove_tags(blog_content[i])

#convert scraped data to usable input form
for i in range(10):
    title_words_count.append(len(ctitle[i].split()))
    contents_words_count.append(len(blog_content[i].split()))
    
for i in range(10):
    print(i)
    words = blog_content[i].split()
    unique_words_count.append(len(" ".join(sorted(set(words), key=words.index)).split()))
    text_tokens = word_tokenize(blog_content[i])
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    
    tokens_without_sw = list(set(tokens_without_sw))
    unique_non_stop_words.append(len(tokens_without_sw))

    
for i in range(0,10):
    catch_links=[]
    res = requests.get(links[i])
    html_page = res.content

    soup =BeautifulSoup(html_page,'html.parser')
    
    for a in soup.find_all('a'):
        content=a.get('href')
        catch_links.append(content)
        length=len(catch_links)
    all_links_count.append(length)

for i in range(10):
    wordlist = blog_content[i].split()
    sum=0
    for word in wordlist:
        sum += len(word)
    avg_length_words.append(sum/contents_words_count[i])
    polarity.append(TextBlob(blog_content[i]).sentiment[0])
    subjectivity.append(TextBlob(blog_content[i]).sentiment[1])
    
sia = SentimentIntensityAnalyzer()

for i in range(10):
    pos=0
    neg=0
    psum=0
    nsum=0
    pos_pol=[]
    neg_pol=[]
    for word in blog_content[i].split():
        if (sia.polarity_scores(word)['compound']) >= 0.5:
            pos += 1
            pos_pol.append(sia.polarity_scores(word)['compound'])
        elif (sia.polarity_scores(word)['compound']) <= -0.5:
            neg += 1
            neg_pol.append(sia.polarity_scores(word)['compound'])
    try:
        avg_pos_pol.append(mean(pos_pol))
    except:
        avg_pos_pol.append(0)
    try:
        avg_neg_pol.append(mean(neg_pol))
    except:
        avg_neg_pol.append(0)
    max_pos_pol.append(max(pos_pol))
    try:
        max_neg_pol.append(max(neg_pol))
    except:
        max_neg_pol.append(0)
    min_pos_pol.append(min(pos_pol))
    try:
        min_neg_pol.append(min(neg_pol))
    except:
        min_neg_pol.append(0)
    pos_words.append(pos)
    neg_words.append(neg)
    
for i in range(10):
    title_pol.append(TextBlob(ctitle[i]).sentiment[0])
    title_sub.append(TextBlob(ctitle[i]).sentiment[1])

#create dummy variables for categorical data "day"   
df_dummy=pd.get_dummies(day)
weekday_is_monday = [0]*10
weekday_is_tuesday = [0]*10
weekday_is_wednesday = df_dummy['Wednesday']
weekday_is_thursday = df_dummy['Thursday']
weekday_is_friday = df_dummy['Friday']
weekday_is_satday = [0]*10
weekday_is_sunday = df_dummy['Sunday']

#convert the lists into a dataframe merging them together
df = pd.DataFrame({'n_tokens_title':title_words_count, 'n_tokens_content':contents_words_count,'n_unique_tokens':unique_words_count,'n_non_stop_unique_tokens':unique_non_stop_words, 'num_hrefs':all_links_count, 'num_imgs':image_count, 'num_videos':video_count, 'average_token_length':avg_length_words, 'num_keywords':num_keywords, 'data_channel_is_lifestyle':data_channel_is_lifestyle, 
                   'data_channel_is_entertainment':data_channel_is_entertainment, 'data_channel_is_bus':data_channel_is_bus,
                   'data_channel_is_socmed':data_channel_is_socmed, 'data_channel_is_tech':data_channel_is_tech, 'data_channel_is_world':data_channel_is_world, 'weekday_is_monday':weekday_is_monday,
                   'weekday_is_tuesday':weekday_is_tuesday, 'weekday_is_wednesday':weekday_is_wednesday, 'weekday_is_thursday':weekday_is_thursday, 'weekday_is_friday':weekday_is_friday,
                   'weekday_is_satday':weekday_is_satday, 'weekday_is_sunday':weekday_is_sunday,
                   'global_subjectivity':subjectivity, 'global_sentiment_polarity':polarity, 'global_rate_positive_words':pos_words, 'global_rate_negative_words':neg_words,
                   'avg_positive_polarity':avg_pos_pol, 'min_positive_polarity':min_pos_pol, 'max_positive_polarity':max_pos_pol, 'avg_negative_polarity':avg_neg_pol, 'min_negative_polarity':min_neg_pol, 'max_negative_polarity':max_neg_pol,
                   'title_subjectivity':title_sub, 'title_sentiment_polarity':title_pol}) 

#scaling of data to be passed as input
scaler = MinMaxScaler()
X=df
X[X.columns] = scaler.fit_transform(X[X.columns])

#load deeplearning model
model = load_model("virality_prediction.h5")
virality = model.predict(X).tolist()
ints = [int(float(num)) for num in virality]
for i in range(10):
    virality[i] = round(virality[i][0])
#convert predictions into dataframe
df1 = pd.DataFrame({'Article Link':clink, 'No of Shares/Likes':virality})


#dataframe to csv
df1.to_csv('viralitypredictionexample.csv',index = False)   
