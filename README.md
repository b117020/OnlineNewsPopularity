# OnlineNewsPopularity
## Problem Statement :   Crawl news & information websites & anticipate the likelihood of its virality. (Bipolar Factory Assignment)

##### Dataset Used for Training :
I have used the [Online News Popularity Data Set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity#) available in the UCI machine learning repository to train my model.

#### Data Set Information:
* The articles were published by Mashable (www.mashable.com) and their content as the rights to reproduce it belongs to them. Hence, this dataset does not share the original content but some statistics associated with it. The original content be publicly accessed and retrieved using the provided urls.
* Acquisition date: January 8, 2015
* The estimated relative performance values were estimated by the authors using a Random Forest classifier and a rolling windows as assessment method. See their article for more details on how the relative performance values were set.

##### Attribute Information:

Number of Attributes: 61 (58 predictive attributes, 2 non-predictive, 1 goal field)

0. url: URL of the article (non-predictive)
1. timedelta: Days between the article publication and the dataset acquisition (non-predictive)
2. n_tokens_title: Number of words in the title
3. n_tokens_content: Number of words in the content
4. n_unique_tokens: Rate of unique words in the content
5. n_non_stop_words: Rate of non-stop words in the content
6. n_non_stop_unique_tokens: Rate of unique non-stop words in the content
7. num_hrefs: Number of links
8. num_self_hrefs: Number of links to other articles published by Mashable
9. num_imgs: Number of images
10. num_videos: Number of videos
11. average_token_length: Average length of the words in the content
12. num_keywords: Number of keywords in the metadata
13. data_channel_is_lifestyle: Is data channel 'Lifestyle'?
14. data_channel_is_entertainment: Is data channel 'Entertainment'?
15. data_channel_is_bus: Is data channel 'Business'?
16. data_channel_is_socmed: Is data channel 'Social Media'?
17. data_channel_is_tech: Is data channel 'Tech'?
18. data_channel_is_world: Is data channel 'World'?
19. kw_min_min: Worst keyword (min. shares)
20. kw_max_min: Worst keyword (max. shares)
21. kw_avg_min: Worst keyword (avg. shares)
22. kw_min_max: Best keyword (min. shares)
23. kw_max_max: Best keyword (max. shares)
24. kw_avg_max: Best keyword (avg. shares)
25. kw_min_avg: Avg. keyword (min. shares)
26. kw_max_avg: Avg. keyword (max. shares)
27. kw_avg_avg: Avg. keyword (avg. shares)
28. self_reference_min_shares: Min. shares of referenced articles in Mashable
29. self_reference_max_shares: Max. shares of referenced articles in Mashable
30. self_reference_avg_sharess: Avg. shares of referenced articles in Mashable
31. weekday_is_monday: Was the article published on a Monday?
32. weekday_is_tuesday: Was the article published on a Tuesday?
33. weekday_is_wednesday: Was the article published on a Wednesday?
34. weekday_is_thursday: Was the article published on a Thursday?
35. weekday_is_friday: Was the article published on a Friday?
36. weekday_is_saturday: Was the article published on a Saturday?
37. weekday_is_sunday: Was the article published on a Sunday?
38. is_weekend: Was the article published on the weekend?
39. LDA_00: Closeness to LDA topic 0
40. LDA_01: Closeness to LDA topic 1
41. LDA_02: Closeness to LDA topic 2
42. LDA_03: Closeness to LDA topic 3
43. LDA_04: Closeness to LDA topic 4
44. global_subjectivity: Text subjectivity
45. global_sentiment_polarity: Text sentiment polarity
46. global_rate_positive_words: Rate of positive words in the content
47. global_rate_negative_words: Rate of negative words in the content
48. rate_positive_words: Rate of positive words among non-neutral tokens
49. rate_negative_words: Rate of negative words among non-neutral tokens
50. avg_positive_polarity: Avg. polarity of positive words
51. min_positive_polarity: Min. polarity of positive words
52. max_positive_polarity: Max. polarity of positive words
53. avg_negative_polarity: Avg. polarity of negative words
54. min_negative_polarity: Min. polarity of negative words
55. max_negative_polarity: Max. polarity of negative words
56. title_subjectivity: Title subjectivity
57. title_sentiment_polarity: Title polarity
58. abs_title_subjectivity: Absolute subjectivity level
59. abs_title_sentiment_polarity: Absolute polarity level
60. shares: Number of shares (target)

##### Methodology Used for training the model

The dataset required some additional cleaning which was done by removing the extra spaces added to column headers and removing the unwanted columns. I had tried using truncatedSVD to the scaled data however the model didn't show much improvement hence i didn't use it in the final version.
I trained the model using various machine learning regression models such as Liner Regression, Ridge, Lasso, BayesianRidge and also applied deep learning to train the model. I compared all the trained models based on their root mean squared errors and mean absolute error and finally selected the deep learning model based on the metrics(mean absolute error). I didn't use accuracy as a metric for selection since it is a regression based model and therefore used mean absolute error as a metric for selection.
The neural network consists of 2 hidden layers along with the input and output layer. All the inputs are scaled using MinMaxScaler and passed into the input layer. I have used batch sizes of 32, 32, 64 for the input, hidden layer 1, hidden layer 2 respectively based on the accuracy of results i got on testing data while training the model.
Finally i saved the model into a h5 file so that it can be used later without training for consistency of results.

##### Methodology used for scraping the data

In order to test my model on real world data I scrapped the required data to be used as features for the model input from [CareerAnna](https://www.careeranna.com/articles/category/mba/pages/) which is a student information site providing information about various technologies and exams . 
I used Selenium and BeautifulSoup for scraping contents from the website.
After scraping the data, it was cleaned and a few NLP operations were performed on it in such that it was converted into a form thatcan be passed as model input along with scaling of the data.
Finally the model(.h5 file) was loaded and the data was given as input and the number of shares/likes was predicted which depicted the virality of the information.An example of the the predicted samples from CareerAnna website can be viewed [here](https://github.com/b117020/OnlineNewsPopularity/blob/master/viralitypredictionexample.csv).
