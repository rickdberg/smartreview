#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:23:32 2018

@author: rick

Script for human-in-the-loop validation of time points
identified by custom NLP algorithm. Allows fast validation and parameter tuning
for individual start and stop times of camera review sections of phone review videos.

Process:
    Run script to show similarity analysis and word occurence plot
    Adjust parameters and buffer to find best fit to start and end times


"""
from __future__ import print_function

import pandas as pd
import numpy as np
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
n_videos = 4
windows = 10

# Get top 5 video urls
fromUser = 'samsung galaxy s9 review'
sql = """select url, title, channel, views, likes, dislikes
            from video_info
            where search_term like "{}"
            and cam_start_time is not null
            order by views*likes/dislikes desc
            limit 5 ;""".format(fromUser)
video_ids = pd.read_sql(sql, engine)


url = video_ids['url'][0]


# Get topics from database
sql = """select *
    from video_topics
    ; """
video_topics = pd.read_sql(sql, con)
cam_words_top = video_topics.iloc[-n_top_words-1:-1,topic_cam]
words_top = " ".join(cam_words_top)

# Similarity analysis
sql = """select *
    from `{}` ;""".format(url)
try:
    transcript = pd.read_sql(sql, engine)
except:
    print('Transcript not found')
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

# Find start time of camera section
cam_occ_times = pd.DataFrame(time_points.loc[camera_counts.nonzero()])
if len(cam_occ_times) == 0:
    print('No camera occurences in {}'.format(url))
cam_similarities = pd.Series(sim_padded)
cam_start_times = cam_occ_times.copy()
cam_start_times['similarities'] = cam_similarities.loc[camera_counts.nonzero()]
cam_start_times = cam_start_times.reset_index()
cam_start_times = cam_start_times.rename(columns = {'index':'idx'})
cam_zero = pd.DataFrame({'idx': [0], 'start_time': [0], 'similarities': [0]})
cam_start_times = pd.concat((cam_zero,cam_start_times)).reset_index(drop=True)
cam_start_options = cam_start_times.iloc[:cam_start_times.similarities[1:].idxmax()+1,:]
cam_delta = pd.Series(np.diff(cam_start_options.start_time))
try:
    cam_delta_idx = cam_delta[cam_delta >= 40].index[-1]
except:
    cam_delta_idx = 0
    print('Only 1 camera occurence in {}'.format(url))
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
    elif cam_sim_options[n + cam_last_idx + 1] == 0 and cam_sim_options[n + cam_last_idx + 2] == 0:
        cam_end_idx = n + cam_last_idx
        break
    elif cam_sim_options[n + cam_last_idx + 1] < 0.6*sim and cam_sim_options[n + cam_last_idx + 2] < 0.6*sim:
        cam_end_idx = n + cam_last_idx
        break
secs_start = time_points[cam_start_idx - 2]
secs_end = time_points[cam_end_idx]

print(url)
print(secs_start/60)
print(secs_end/60)

# Create figure
figure, ax1 = plt.subplots(1, 1, figsize=(9, 3))
#figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5))
# ax1.plot(time_points/60, similarities[:,topic_ux], label="Topic "+str(topic_ux))
# ax1.plot(time_points/60, similarities[:,topic_sound], label="Topic "+str(topic_sound))


ax1.stem(time_points[camera_counts != 0]/60, camera_counts[camera_counts != 0], 'purple')
ax1.set_ylim([0,max(sim_padded)])

# ax2.plot(time_points/60, word_counts[:,topic_cam], label="Topic "+str(topic_cam))
# ax2.plot(time_points/60, word_counts[:,topic_ux], label="Topic "+str(topic_ux))
# ax2.plot(time_points/60, word_counts[:,topic_sound], label="Topic "+str(topic_sound))
# ax2.plot(time_points/60, camera_counts, color='k')

ax1.fill_betweenx([0,max(sim_padded)],secs_start/60,secs_end/60,color='purple', alpha=0.5)
ax1.plot(time_points/60, sim_padded, label="Cosine similarity", color='dodgerblue')
ax1.plot(time_points[cam_start_idx - 2:cam_end_idx+1]/60, sim_padded[cam_start_idx - 2:cam_end_idx+1], label="Cosine similarity", color='cyan')

# ax2.fill_betweenx([0,max(similarities[:,topic_cam])],secs_start/60,secs_end/60,color='blue', alpha=0.4)
#ax1.set_ylim([0,np.max(sim_padded)+0.01])
ax1.set_xlim([0,np.max(time_points/60)])
ax1.locator_params(axis='x', nbins=5)
ax1.locator_params(axis='y', nbins=5)
ax1.set_xlabel("$Minutes$", fontsize=20)
ax1.set_ylabel("$Cosine\ similarity$", fontsize=20, color='dodgerblue')
ax1.tick_params(labelsize=18)
#ax2.set_ylim([0,np.max(word_counts)+0.1])
#ax2.set_xlim([0,np.max(time_points/60)])
#ax2.locator_params(axis='x', nbins=5)
#ax2.locator_params(axis='y', nbins=5)
#ax2.set_ylabel("$Occurences$", fontsize=20)
#ax2.set_xlabel("$Minutes$", fontsize=20)

#ax1.legend(loc=(1.04,0))
figure.tight_layout()
# figure.savefig('video_fig.png')
plt.show()


# eof
