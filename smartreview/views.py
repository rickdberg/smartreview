from __future__ import print_function

from flask import render_template, send_file
from smartreview import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
from flask import request
from time_bar import plot_bar
import io

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


user = 'root' #add your username here (same as previous postgreSQL)                      
host = '127.0.0.1'
dbname = 'insight'
engine = create_engine('mysql+mysqldb://%s:backcountry@%s/%s'%(user,host,dbname))
con = None
con = engine.connect()

@app.route('/')
@app.route('/index')
def index():
    return render_template("input.html",
        )

@app.route('/db')
def birth_page():
    sql_query = """                                                                       
                SELECT * FROM video_info WHERE channel='The Verge';          
                """
    query_results = pd.read_sql_query(sql_query,con)
    
    return query_results.iloc[0,2] + "<br>"

@app.route('/db_fancy')
def cesareans_page_fancy():
    sql_query = """
               SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    query_results=pd.read_sql_query(sql_query,con)
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
    return render_template('cesareans.html',births=births)

@app.route('/output')
def cesareans_output():
    #pull 'fromUser' from input field and store it
    fromUser = request.args.get('fromUser')
    fig_name = plot_bar(fromUser)
    video_fig  = '/static/figs/'+fig_name
    return render_template("output.html", fromUser = fromUser, video_fig = video_fig)

@app.route('/input')
def cesareans_input():
    return render_template("input.html")
