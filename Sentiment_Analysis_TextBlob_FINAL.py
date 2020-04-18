
# Basic Twitter comment sentiment analysis using NLTK and TextBlob

import pandas as pd
import matplotlib.pyplot as plt
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from textblob import TextBlob

# Remove regular expressions

def remove_re(tweet):

    cleaned = []
    for t in tweet:
        cleaned.append(' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", t).split()))

    return cleaned

# Remove stopwords

def remove_stopwords(tweet):

    stopwords_set = set(stopwords.words('english'))
    tweets_without_stopwords = []
    for index, row in enumerate(tweet):
        words_filtered = [e.lower() for e in row.split() if len(e) >= 3]
        words_cleaned = [word for word in words_filtered 
                        if 'http' not in word 
                        and not word.startswith('@') 
                        and not word.startswith('#') 
                        and word != 'RT']
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
        final = ' '.join(words_without_stopwords)
        tweets_without_stopwords.append(final)

    return tweets_without_stopwords

# Get sentiment polarity

def get_tweet_sentiment(analisys):

    if analisys.sentiment.polarity > 0:
        return 'positive'
    elif analisys.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Get words from each tweet

def get_words_tweets(stringlist):

    sentence_split = [sentence.split() for sentence in stringlist]
    words = [word for sublist in sentence_split for word in sublist]
    for w in words:
        word_tokenize(w)

    return words

# Discriminate words by class (only select nouns and adjectives)

def allowed_tweets(tweets):

    allowed_words_types = ['NN', 'NNS', 'JJ']
    w_tag = nltk.pos_tag(get_words_tweets(tweets))
    allowed_tweets = [w[0] for i in allowed_words_types for w in w_tag if w[1] in allowed_words_types]

    return allowed_tweets

# Open data and apply functions

data_tweet = pd.read_csv('GOP_REL_ONLY.csv', encoding = 'ISO-8859-1')
tweet_text = data_tweet['text']

cleaned_tweet = remove_re(data_tweet['text'])
final_tweet = remove_stopwords(cleaned_tweet)

sentiment = []
for t in final_tweet:  
    sentiment.append(get_tweet_sentiment(TextBlob(t)))
    
# Percentage of positive, neutral and negative tweets

tweet_pos = [final_tweet[i] for i in range(len(sentiment)) if sentiment[i] == 'positive']
print("Positive tweets: {} %".format(100*len(tweet_pos)/len(sentiment)))
tweet_neu = [final_tweet[i] for i in range(len(sentiment)) if sentiment[i] == 'neutral']
print("Neutral tweets: {} %".format(100*len(tweet_neu)/len(sentiment)))
tweet_neg = [final_tweet[i] for i in range(len(sentiment)) if sentiment[i] == 'negative']
print("Negative tweets: {} %".format(100*len(tweet_neg)/len(sentiment)))

# Appearance of the candidates in positive tweets

freq_pos = nltk.FreqDist(allowed_tweets(tweet_pos))
freq_neg = nltk.FreqDist(allowed_tweets(tweet_neg))

can = [freq_pos['trump'], freq_pos['cruz'], freq_pos['rubio'], 
        freq_pos['kasich'], freq_pos['carson'], freq_pos['bush'], 
        freq_pos['paul'], freq_pos['huckabee'], freq_pos['fiorina']]
lab = ['Trump', 'Cruz', 'Rubio', 'Kasich', 'Carson', 'Bush', 'Paul', 'Huckabee', 'Fiorina']

plt.pie(can, labels=lab, startangle=90, autopct='%1.1f%%')
plt.show()

# Trump's sentiment based on his appearance in positive and negative tweets

div = [freq_pos['trump'], freq_neg['trump']]
lab = ['Trump s positive comments', 'Trump s negative comments']
plt.pie(div, labels=lab, startangle=90, autopct='%1.1f%%')
print(plt.show())