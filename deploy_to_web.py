#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 19:47:08 2018

@author: rick

Read video analysis results from AWS RDS, define variables for web app
at smartcamreview.com

"""

import numpy as np
import pandas as pd
import re
import sqlalchemy as sa

from access import db_user, db_pw, db_path

def video_data(fromUser = 'Default'):

    if fromUser == 'Default':
        fromUser = 'samsung galaxy s9'

    vid_num = 9

    engine = sa.create_engine("mysql+mysqldb://{}:{}@{}:3306/insight?charset=utf8".format(db_user, db_pw, db_path), encoding='utf-8')
    con = engine.connect()

    fromUser = fromUser + ' review'
    sql = """select *
    from video_info
    where cam_start_time is not null
    and search_term like '{}'
    ;""".format(fromUser)
    video_info = pd.read_sql(sql, con)

    video_info.views = video_info.views.apply(pd.to_numeric)
    video_info = video_info.sort_values('views', axis=0, ascending=False, inplace=False)
    video_info = video_info.reset_index(drop=True)

    try:
        video_info = video_info.iloc[:vid_num,:]
    except:
        video_info = video_info.iloc[:,:]
    # Video lengths
    minutes = []
    seconds = []
    len_full = []
    video_lengths = video_info['length']
    video_info = video_info[~video_lengths.str.contains('H')]
    for n in video_info['length']:
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


    ends = video_info['cam_end_time'].astype(float)
    # ends_act = video_info['cam_end_time_actual'].astype(float)

    starts = video_info['cam_start_time'].astype(float)
    starts = np.floor(starts).astype(int)
    # starts_act = video_info['cam_start_time_actual'].astype(float) + 1
    # durations = ends - starts

    urls = video_info['url']
    channels = video_info['channel']
    views = video_info['views']
    titles = video_info['title']

    cam_occs = []
    for url in urls:
        sql = '''select start_time
            from `{}_cam_times`
            ;'''.format(url)
        occs = pd.read_sql(sql, engine)
        occs = pd.Series(occs['start_time']).tolist()
        cam_occs.append(occs)

    return starts, ends, cam_occs, urls, channels, views, titles

