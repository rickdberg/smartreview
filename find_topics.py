#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:10:00 2018

@author: rick

Do stats for presentation
    Histogram of length of videos
    Histogram of length of popular videos

    Histogram of length of videos and length of camera sections
    Histogram of length of videos and length of user interface sections



"""
from __future__ import print_function

import spacy
import nltk
import re
import pandas as pd
import sqlalchemy as sa
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en import English
from sklearn.decomposition import NMF
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn

from access import db_root_pw

n_samples = 500
n_features = 1200
n_components = 15
n_top_words = 15
windows = 10
topic_cam = 1
topic_ux = 5
topic_sound = 4
product_set = ('iphone', 'apple','plus','galaxy',' 10','s9',' 8','samsung',
               '6s',' 7',' new','android','ios','upgrade','pixel',' 2','xl',
               'google','huawei','p20','china','chinese',' mate',' pro',' note',
               'j7','lg','g7','thinq','oneplus',' 5t',' 5',' 6','sony','xperia',
               's8','phone','mobile','xz2','emoji','super','bixby','watch','quite',
               'easy','result','easie')

engine = sa.create_engine("mysql+mysqldb://root:{}@127.0.0.1/insight?charset=utf8".format(db_root_pw), encoding='utf-8')
con = engine.connect()

# nltk.download('wordnet')

sql = """select * from video_info ;"""
video_info = pd.read_sql(sql, engine)
video_info_filtered = video_info[~video_info['title'].str.contains('vs')]
video_info_filtered = video_info_filtered[~video_info_filtered['title'].str.contains('Unbox')]
video_info_filtered = video_info_filtered[~video_info_filtered['title'].str.contains('unbox')]
video_info_filtered = video_info_filtered[~video_info_filtered['title'].str.contains('vs')]
video_info_filtered = video_info_filtered[~video_info_filtered['title'].str.contains('Vs')]
video_info_filtered = video_info_filtered[~video_info_filtered['channel'].str.contains('Vs')]
urls_filtered = pd.Series(video_info_filtered['url'])

sql = """select length
            from video_info
            ;"""
video_all = pd.read_sql(sql, engine)

# Compile video transcripts
transcript_data = []
for n in urls_filtered:
    sql = """select caption from `{}` ;""".format(n)
    try:
        vid_captions = pd.read_sql(sql, engine)
        vid_captions = pd.Series(vid_captions.iloc[:,0])
        vid_captions = vid_captions.tolist()
    except:
        continue
    full_string = ' '.join(vid_captions)
    for word in product_set:
        try:
            insensitive_word = re.compile(re.escape(word), re.IGNORECASE)
            full_string = insensitive_word.sub('', full_string)
        except:
            continue
    transcript_data.append(full_string)




## Tokenize, lemmatize, filter stop words
spacy.load('en')
parser = English()
# nltk.download('stopwords')

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

# Tokenizer
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

# Lemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

# Define stop words
en_stop = set(nltk.corpus.stopwords.words('english'))

# Preparer function for topic modeling
def prepare_text_for_tm(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

# Print selected number of top words
def print_top_words(model, feature_names, n_top_words):
    topic_list = pd.DataFrame()
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        topic_list["topic_%i" % topic_idx] = [feature_names[i]
                             for i in topic.argsort()]
        print(message)
    print()
    return topic_list


# Prep data for topic modeling
text_data = []
for line in transcript_data:
    tokens = prepare_text_for_tm(line)
    text_data.append(tokens)

# Create full bag of words from token lists
print("Loading dataset...")
text_bag = []
for line in text_data:
    bag = ' '.join(line)
    text_bag.append(bag)
data_samples = text_bag # dataset.data[:n_samples]
t0 = time()
print("done in %0.3fs." % (time() - t0))

# Use tf-idf features for NMF
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=0.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
topic_list = print_top_words(nmf, tfidf_feature_names, n_top_words)


# Send topics to sql as a table







# Count total words used in topic model fit
all_words = ''
for n in transcript_data:
    all_words +=  str(n)
len(all_words.split())

# Turn time stamps into total seconds and plot all video lengths
minutes = []
seconds = []
total_times = []
video_all = video_all[~video_all.iloc[:,0].str.contains('H')]
for n in video_all.iloc[:,0]:
    lengths = re.split('T',n)
    mins = re.split('M',lengths[1])
    just_mins = mins[0]
    minutes.append(just_mins)
    secs = mins[1]
    just_secs = re.split('S', secs)[0]
    if just_secs == '':
        just_secs = 0
    seconds.append(just_secs)
    total_tim = float(just_mins)+float(just_secs)/60
    total_times.append(total_tim)
plt.hist(total_times, bins = 100)

