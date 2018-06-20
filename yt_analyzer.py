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
#import random
import nltk
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en import English
from sklearn.decomposition import NMF
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
import sqlalchemy as sa

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


fromUser = 'iphone x review'
sql = """select url, title, channel, views, likes, dislikes
            from video_info
            where search_term like "{}"
            order by views*likes/dislikes desc
            limit 20 ;""".format(fromUser)
video_ids = pd.read_sql(sql, engine)
video_ids = video_ids[~video_ids['title'].str.contains('vs')]
video_ids = video_ids[~video_ids['title'].str.contains('Unbox')]
video_ids = video_ids[~video_ids['channel'].str.contains('Unbox')]

video_id = video_ids.iloc[5][0]

sql = """select length
            from video_info
            ;""".format(fromUser)
video_all = pd.read_sql(sql, engine)

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

sql = """select *
    from {} ;""".format(video_id)
transcript = pd.read_sql(sql, engine)

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

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

# Filter out stop words
en_stop = set(nltk.corpus.stopwords.words('english'))

# Prepare for LDA
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

# View tokens
text_data = []
for line in transcript_data:
    tokens = prepare_text_for_lda(line)
    text_data.append(tokens)
    #if random.random() > .99:

text_bag = []
for line in text_data:
    bag = ' '.join(line)
    text_bag.append(bag)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

print("Loading dataset...")
t0 = time()
data_samples = text_bag  # dataset.data[:n_samples]
print("done in %0.3fs." % (time() - t0))

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")  #TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=3,
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
          alpha=.1, l1_ratio=.5).fit(tfidf)  #Change to tfidf for tf-idf
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

snippets = transcript.caption
time_points = pd.to_timedelta(transcript.start_time)
time_points = time_points.dt.total_seconds()
word_counts = np.zeros([len(snippets),n_components])
similarities = np.zeros([len(snippets),n_components])
for topic_idx, topic in enumerate(nmf.components_):
    component_words = " ".join([tfidf_feature_names[i]
                     for i in topic.argsort()[:-n_top_words - 1:-1]])
    comp_word_list = component_words.replace(' ',',').split(",")
    block = int(np.floor(len(snippets)/windows))
    camera_counts = np.zeros(len(snippets))
    for n in np.arange(len(snippets)-block):
        camera_count = snippets[n].count('camera')
        camera_counts[n] = camera_count

        snippet_block = ' '.join(snippets.iloc[n:n+block])
        tfidf = TfidfVectorizer().fit_transform([component_words,snippet_block])
        pairwise_similarity = tfidf * tfidf.T
        similarities[n,topic_idx] = pairwise_similarity[0,1]

        word_count = 0
        for word in comp_word_list:
            word_count += snippet_block.count(word)
        word_counts[n, topic_idx] = word_count

# Find start time of camera section
cam_start_times = pd.DataFrame(time_points.loc[camera_counts.nonzero()])
cam_similarities = pd.Series(similarities[:,topic_cam])
cam_start_times['similarities'] = cam_similarities.loc[camera_counts.nonzero()]
cam_start_times = cam_start_times.reset_index()
cam_start_times = cam_start_times.rename(columns = {'index':'idx'})
start_options = cam_start_times.iloc[:cam_start_times.similarities.idxmax()+1,:]
cam_delta = pd.Series(np.diff(start_options.start_time))
cam_delta_idx = cam_delta[cam_delta >= 30].index[-1]
cam_start_time = cam_start_times.start_time.iloc[cam_delta_idx+1]
cam_start_idx = cam_start_times.idx.iloc[cam_delta_idx+1]

# Find end time of camera section
cam_sim_options = cam_similarities[cam_start_idx:]
for n, sim in enumerate(cam_sim_options):
    if sim == 0 and cam_sim_options[n + cam_start_idx + 1] == 0:
        cam_end_idx = n + cam_start_idx - 1
        break

secs_start = time_points[cam_start_idx - 2]
secs_end = time_points[cam_end_idx + 1]

# Create figure
figure, (ax1,ax2) = plt.subplots(2, 1, figsize=(9, 6.5))
#figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5))
ax1.plot(time_points/60, similarities[:,topic_cam], label="Topic "+str(topic_cam))
# ax1.plot(time_points/60, similarities[:,topic_ux], label="Topic "+str(topic_ux))
# ax1.plot(time_points/60, similarities[:,topic_sound], label="Topic "+str(topic_sound))
ax1.plot(time_points/60, camera_counts, color='k')

ax2.plot(time_points/60, word_counts[:,topic_cam], label="Topic "+str(topic_cam))
# ax2.plot(time_points/60, word_counts[:,topic_ux], label="Topic "+str(topic_ux))
# ax2.plot(time_points/60, word_counts[:,topic_sound], label="Topic "+str(topic_sound))
ax2.plot(time_points/60, camera_counts, color='k')

ax1.fill_betweenx([0,max(similarities[:,topic_cam])],secs_start/60,secs_end/60,color='blue', alpha=0.4)
ax2.fill_betweenx([0,max(similarities[:,topic_cam])],secs_start/60,secs_end/60,color='blue', alpha=0.4)
ax1.set_ylim([0,np.max(similarities)+0.01])
ax1.set_xlim([0,np.max(time_points/60)])
ax1.locator_params(axis='x', nbins=5)
ax1.locator_params(axis='y', nbins=5)
ax1.set_xlabel("$Minutes$", fontsize=20)
ax1.set_ylabel("$Cosine\ similarity$", fontsize=20)

ax2.set_ylim([0,np.max(word_counts)+0.1])
ax2.set_xlim([0,np.max(time_points/60)])
ax2.locator_params(axis='x', nbins=5)
ax2.locator_params(axis='y', nbins=5)
ax2.set_ylabel("$Occurences$", fontsize=20)
ax2.set_xlabel("$Minutes$", fontsize=20)

ax1.legend(loc=(1.04,0))
figure.tight_layout()
# figure.savefig('video_fig.png')

all_words = ''
for n in transcript_data:
    all_words +=  str(n)
len(all_words.split())

