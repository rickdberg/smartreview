#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:10:08 2018

@author:
"""

# import json
from apiclient.discovery import build
# from apiclient.errors import HttpError
# from oauth2client.tools import argparser
import os
import re
from webvtt import WebVTT
import pandas as pd
import numpy as np
import sqlalchemy as sa

from access import db_root_pw, google_dev_key

rev_num = 20  # Number of videos, up to 50
search = 'iphone x review'  # YouTube search


engine = sa.create_engine("mysql+mysqldb://root:{}@127.0.0.1/insight?charset=utf8".format(db_root_pw), encoding='utf-8')
con = engine.connect()
# con.set_character_set('utf8')
# https://developers.google.com/youtube/v3/code_samples/python
# https://github.com/spnichol/youtube_tutorial/blob/master/youtube_videos.py
# Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
# tab of
#   https://cloud.google.com/console
# Please ensure that you have enabled the YouTube Data API for your project.
DEVELOPER_KEY = google_dev_key    #  Your developer key here
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def youtube_search(q, max_results=rev_num, order="relevance",token=None,location=None,location_radius=None):
  youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
    developerKey=DEVELOPER_KEY)

  # Call the search.list method to retrieve results matching the specified
  # query term.
  search_response = youtube.search().list(
    q=q,
    type="video",
    pageToken=token,
    order=order,
    part="id,snippet",
    maxResults=max_results,
    location=location,
    locationRadius=location_radius
  ).execute()

  videos = []
#  channels = []
#  playlists = []

  # Add each result to the appropriate list, and then display the lists of
  # matching videos, channels, and playlists.
  for search_result in search_response.get("items", []):
    if search_result["id"]["kind"] == "youtube#video":
      videos.append(search_result)
  try:
      nexttok = search_response["nextPageToken"]
      return(nexttok, videos)
  except Exception as e:
      nexttok = "last_page"
      return(nexttok, videos)

def geo_query(video_id):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                    developerKey=DEVELOPER_KEY)
    video_response = youtube.videos().list(
        id=video_id,
        part='snippet, recordingDetails, statistics, contentDetails,  player'
    ).execute()
    return video_response


review_list = youtube_search("{}".format(search))
just_json = review_list[1]
yt_urls = []
titles = []
channels = []
for video in just_json:
    yt_urls.append(video['id']['videoId'])
    titles.append(video['snippet']['title'])
    channels.append(video['snippet']['channelTitle'])


def find_between(s, first="Read More: ", last="com" ):
    try:
        start = s.index(first) + len(first)
        end = s.index( last, start )
        return s[start:end]+"com"
    except ValueError:
        return None

total_length = []
views = []
likes = []
dislikes = []
descriptions = []
comments = []
for vid_url in yt_urls:
    video_info = geo_query(vid_url)
    total_length.append(video_info['items'][0]['contentDetails']['duration'])
    views.append(video_info['items'][0]['statistics']['viewCount'])
    likes.append(video_info['items'][0]['statistics']['likeCount'])
    dislikes.append(video_info['items'][0]['statistics']['dislikeCount'])
    comments.append(video_info['items'][0]['statistics']['commentCount'])
    description = video_info['items'][0]['snippet']['localized']['description']
    descriptions.append(find_between(description, "Read More: ", "com"))

search_term = [search] * len(yt_urls)
video_table = pd.DataFrame({"url":yt_urls, "title":titles, "channel":channels, "length":total_length, "views":views, "likes":likes, "dislikes":dislikes, "comments":comments, "description_info":descriptions, "search_term":search_term})
sql = """select * from video_info ;"""
# video_sql = pd.read_sql(sql, engine)
# join_cols = ['url','title','channel','length','views','likes','dislikes','comments','description_info','search_term']
# video_export = pd.merge(video_sql, video_table, how='outer', on=join_cols)

# video_export.to_sql('video_info', engine, if_exists='replace', index=False)


for n in np.arange(len(yt_urls)):
    inputs = (yt_urls[n], titles[n], channels[n], total_length[n], views[n], likes[n], dislikes[n], comments[n], descriptions[n], search_term[n], views[n], likes[n], dislikes[n], comments[n], search_term[n])
    sql= """insert into video_info (url,title,channel,length,views,likes,dislikes,comments,description_info,search_term)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE views=%s,likes=%s,dislikes=%s,comments=%s,search_term=%s
                ;"""
    con.execute(sql, inputs)

# https://stackoverflow.com/questions/48125300/cant-scrape-youtube-videos-closed-captions
#I'll need to autopopulate the output filenames with %
def download_subs(video_url, lang="en"):
    cmd = [
        "youtube-dl",
        "--skip-download",
        "--write-auto-sub",
        "--sub-format vtt",
        "--sub-lang en",
        video_url,
        "--output test.vtt"
    ]

    command_string = " ".join(cmd)
    print(command_string)
    os.system(command_string)
# https://pypi.org/project/webvtt-py/

for j, yt_url in enumerate(yt_urls):
    url = "https://www.youtube.com/watch?v="+yt_url
    os.system("rm test.en.vtt")
    print(j+1, " out of ",rev_num)
    vtt_test = download_subs(url)
    try:
        vtt_obj = WebVTT().read('test.en.vtt')
    except:
        continue
    transcript = pd.DataFrame(np.empty((len(vtt_obj),3)), columns=['start_time','end_time','caption'])
    for n, caption in enumerate(vtt_obj):
        transcript.iloc[n,0] = caption.start
        transcript.iloc[n,1] = caption.end
        transcript.iloc[n,2] = caption.text
    # Clean up the transcripts that have duplicate entries
    unique_transcript_idx = np.zeros(len(transcript))
    for m, record in enumerate(transcript['caption']):
        phrase = transcript.iloc[m,2]
        phrase = phrase.strip()
        phrase = re.sub(r"\-", " ", phrase)
        phrase = re.sub(r"\+", "plus", phrase)
        phrase = re.sub(r"\n", " ", phrase)
        if (phrase == '' or phrase == "[Music]" or phrase == "[Applause]" or phrase == "[Music][Applause]"):
            unique_transcript_idx = np.zeros(len(transcript))
        else:
            unique_transcript_idx = transcript['caption'].str.contains("%(ph)s" % {'ph': phrase})
        for idx, tf in enumerate(unique_transcript_idx):
            if tf == True:
                transcript.iloc[idx,2] = phrase
    transcript = transcript.drop_duplicates('caption').reset_index(drop=True)
    for i in np.arange(len(transcript['end_time'])-1):
        transcript.iloc[i,1] = transcript.iloc[i+1,0]
    transcript.to_sql(yt_url, engine, if_exists='replace', index=False)

