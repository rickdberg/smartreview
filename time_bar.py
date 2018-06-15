#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 19:37:05 2018

@author: rick
"""

from __future__ import print_function

import spacy
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


def plot_bar(fromUser  = 'Default'):

    if fromUser == 'Default':
        fromUser = 'samsung galaxy s9 review'

    n_samples = 2000
    n_features = 1000
    n_components = 9
    n_top_words = 16

    product_set = ('iphone', 'apple','plus','galaxy',' 10','s9',' 8','samsung',
                   '6s',' 7',' new','android','ios','upgrade','pixel',' 2','xl',
                   'google','huawei','p20','china','chinese',' mate',' pro',' note',
                   'j7','lg','g7','thinq','oneplus',' 5t',' 5',' 6','sony','xperia',
                   's8','phone','mobile','xz2')

    engine = sa.create_engine("mysql+mysqldb://root:backcountry@127.0.0.1/insight?charset=utf8", encoding='utf-8')
    con = engine.connect()

    # nltk.download('wordnet')

    sql = """select * from video_info ;"""
    video_info = pd.read_sql(sql, engine)
    video_info_filtered = video_info[~video_info['title'].str.contains('vs')]
    video_info_filtered = video_info_filtered[~video_info_filtered['title'].str.contains('Unbox')]
    video_info_filtered = video_info_filtered[~video_info_filtered['title'].str.contains('unbox')]

    urls_filtered = pd.Series(video_info_filtered['url'])

    sql = """select url
                from video_info
                where search_term like "{}"
                order by views desc
                limit 1 ;""".format(fromUser)
    video_id = pd.read_sql(sql, engine)
    video_id = video_id.iloc[0][0]

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
              alpha=.1, l1_ratio=.5).fit(tfidf)  #Change to tfidf for tf-idf
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    # Create figure
    video_fig, ax1 = plt.subplots(1, 1, figsize=(9, 3))
    time_points = pd.to_timedelta(transcript.iloc[:,0])
    time_points = time_points.dt.total_seconds()
    word_counts = np.zeros([len(transcript),n_components])
    similarities = np.zeros([len(transcript),n_components])
    for topic_idx, topic in enumerate(nmf.components_):
        component_words = " ".join([tfidf_feature_names[i]
                         for i in topic.argsort()[:-n_top_words - 1:-1]])
        comp_word_list = component_words.replace(' ',',').split(",")
        snippets = transcript.caption
        block = int(np.floor(len(snippets)/10))
        for n in np.arange(len(snippets)-block):
            snippet_block = ' '.join(snippets.iloc[n:n+block])
            tfidf = TfidfVectorizer().fit_transform([component_words,snippet_block])
            pairwise_similarity = tfidf * tfidf.T
            similarities[n,topic_idx] = pairwise_similarity[0,1]
            word_count = 0
            for word in comp_word_list:
                word_count += snippet_block.count(word)
            word_counts[n, topic_idx] = word_count
        #ax2.plot(time_points/60, word_counts[:,topic_idx], label="Topic "+str(topic_idx))
        ax1.plot(time_points/60, similarities[:,topic_idx], label="Topic "+str(topic_idx))

    #if fromUser == 'samsung galaxy s9 review':
       # ax1.fill_betweenx([0,np.max(word_counts)],2.75,6.75,color='blue', alpha=0.4)
    ax1.set_ylim([0,np.max(similarities)+0.01])
    ax1.set_xlim([0,np.max(time_points/60)])
    ax1.locator_params(axis='x', nbins=5)
    ax1.locator_params(axis='y', nbins=5)
    ax1.set_xlabel("$Minutes$", fontsize=20)
    ax1.set_ylabel("$Cosine\ similarity$", fontsize=20)



    ax1.legend(loc=(1.04,0))
    video_fig.tight_layout()
    
    rand_suffix = str(int(np.floor(np.random.rand(1)*10000)))

    fig_name = 'video_fig_'+rand_suffix+'.png'
    video_fig.savefig('/home/rick/Git/insight/smartreview/static/figs/'+fig_name)
    return fig_name
"""
    ax2.set_ylim([0,np.max(word_counts)+0.1])
    ax2.set_xlim([0,np.max(time_points/60)])
    ax2.locator_params(axis='x', nbins=5)
    ax2.locator_params(axis='y', nbins=5)
    ax2.set_ylabel("$Occurences$", fontsize=20)
    ax2.set_xlabel("$Minutes$", fontsize=20)
    ax2.fill_betweenx([0,np.max(word_counts)],2.75,6.75,color='blue', alpha=0.4)
"""
