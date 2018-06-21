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


    apply sentiment to window
    apply sentiment to full
    send info to db, replace

"""
from __future__ import print_function

import pandas as pd
import numpy as np
import re
import sqlalchemy as sa
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from access import db_root_pw

engine = sa.create_engine("mysql+mysqldb://root:{}@127.0.0.1/insight?charset=utf8".format(db_root_pw), encoding='utf-8')
con = engine.connect()

topic_cam = 1
topic_ux = 5
topic_sound = 4
n_top_words = 15
windows = 10


# Get topics from database
sql = """select *
    from video_topics
    ; """
video_topics = pd.read_sql(sql, con)
cam_words_top = video_topics.iloc[-n_top_words-1:-1,topic_cam]
words_top = " ".join(cam_words_top)

sql = """select *
    from video_info
    ;"""         # Change when doing sentiment where cam_start_time is Null
video_info = pd.read_sql(sql, engine)
video_urls = pd.Series(video_info['url'])


for url in video_urls:
    sql = """select *
        from `{}` ;""".format(url)
    try:
        transcript = pd.read_sql(sql, engine)
    except:
        print('Transcript ' +'{}'.format(url) + ' not found')
        continue
    snippets = transcript.caption
    time_points = pd.to_timedelta(transcript.start_time)
    time_points = time_points.dt.total_seconds()

    similarities = np.zeros(len(snippets))
    block = int(np.floor(len(snippets)/(windows)))
    sim_pad = np.zeros(block)
    sim_padded = similarities
    snip_pad = np.empty(block,dtype=str)
    snip_padded = np.concatenate((snippets.tolist(),snip_pad))
    camera_counts = np.zeros(len(snippets))
    for n in np.arange(len(snippets)):
        camera_count = snip_padded[n].count('camera')
        camera_counts[n] = camera_count

        snippet_block = ' '.join(snip_padded[n:n+block])
        tfidf = TfidfVectorizer().fit_transform([words_top,snippet_block])
        pairwise_similarity = tfidf * tfidf.T
        sim_padded[n] = pairwise_similarity[0,1]

        # Sentiment analysis here


    # Find start time of camera section
    cam_occ_times = pd.DataFrame(time_points.loc[camera_counts.nonzero()])
    cam_similarities = pd.Series(sim_padded)
    cam_start_times = cam_occ_times.copy()
    cam_start_times['similarities'] = cam_similarities.loc[camera_counts.nonzero()]
    cam_start_times = cam_start_times.reset_index()
    cam_start_times = cam_start_times.rename(columns = {'index':'idx'})
    cam_zero = pd.DataFrame({'idx': [0], 'start_time': [0], 'similarities': [0]})
    cam_start_times = pd.concat((cam_zero,cam_start_times)).reset_index(drop=True)

    start_options = cam_start_times.iloc[:cam_start_times.similarities[1:].idxmax()+1,:]
    cam_delta = pd.Series(np.diff(start_options.start_time))
    try:
        cam_delta_idx = cam_delta[cam_delta >= 40].index[-1]
    except:
        print('Not enough camera occurences')
    cam_start_time = cam_start_times.start_time.iloc[cam_delta_idx+1]
    cam_start_idx = cam_start_times.idx.iloc[cam_delta_idx+1]

    # Find end time of camera section
    cam_highest_idx = cam_start_times.idx[cam_start_times.similarities[1:].idxmax()]
    cam_end_options = cam_start_times.iloc[cam_start_times.similarities[1:].idxmax():,:]
    cam_delta_end = pd.Series(np.diff(cam_end_options.start_time))
    if len(cam_delta_end) == 0:
        cam_last_idx = cam_highest_idx
    else:
        try:
            cam_delta_idx_end = cam_delta_end[cam_delta_end >= 44].index[0] + cam_start_times.similarities[1:].idxmax()
        except:
            cam_delta_idx_end = cam_delta_end.index[-1] + cam_start_times.similarities[1:].idxmax()
            print('Last camera occurence')
        cam_last_time = cam_end_options.start_time[cam_delta_idx_end]
        cam_last_idx = cam_end_options.idx[cam_delta_idx_end]

    cam_sim_options = cam_similarities[cam_last_idx:]
    for n, sim in enumerate(cam_sim_options):
        if n == len(cam_sim_options)-1:
            cam_end_idx = n + cam_last_idx
            break
        elif n == len(cam_sim_options)-2:
            cam_end_idx = n + cam_last_idx
            break
        elif cam_sim_options[n + cam_last_idx + 1] < 0.6*sim and cam_sim_options[n + cam_last_idx + 2] < 0.6*sim:
            cam_end_idx = n + cam_last_idx
            break
    secs_start = time_points[cam_start_idx - 2]
    secs_end = time_points[cam_end_idx]

    # Add start/end times to database
    inputs = (url,secs_start,secs_end,secs_start,secs_end)
    sql = """insert into video_info (url,cam_start_time,cam_end_time)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE cam_start_time=%s,cam_end_time=%s
                ;"""
    con.execute(sql, inputs)

    # Add camera_occ_times to database
    cam_occ_times.to_sql('{}'.format(url)+'_cam_times', con, if_exists='replace')


# Make histograms
sql = """select *
    from video_info
    where cam_start_time is not null;"""
video_info = pd.read_sql(sql, engine)

minutes = []
seconds = []
len_full = []
video_lengths = video_info['length']
video_lengths = video_info[~video_lengths.str.contains('H')]
for n in video_lengths['length']:
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
    len_full.append(total_tim)


len_cam = video_lengths['cam_end_time'].astype(float)/60 - video_lengths['cam_start_time'].astype(float)/60
# plt.hist(len_cam, bins=100, alpha=0.5)
# plt.hist(len_full, bins=100, alpha=0.5)

plt.hist(len_full-len_cam, bins=100)

# eof
