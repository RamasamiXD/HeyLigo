
from flask import Flask ,render_template, json, request, jsonify , make_response, Response
app = Flask(__name__)
import matplotlib
matplotlib.use('WebAgg')
from sklearn.preprocessing import MinMaxScaler
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import random
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import os, seaborn
from functools import wraps
import math
import pickle
import operator
import requests
import json
import random
import subprocess
hlusername = 'heyLigo' #os.environ.get('hlusername')
hlpassword = 'iucaa123' #os.environ.get('hlpassword')
num_feat = 500
number_of_results = 30
def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return username == hlusername and password == hlpassword

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

def get_senti_colors(pro_df):
    cs = pd.read_csv('Senti_words.csv',index_col=0)
    word_set = set(cs.index.values)
    word_dict = cs.to_dict()['score']

    def senti(x,word_dict=word_dict,word_set=word_set):
        x = str(x).split()
        val =  [word_dict[i] for i in x if i in word_set]
        if val:
            val =  sum(val)/len(val)
        else:
            val =  0
        return val

    pro_df['title_authors_stems_sentiments'] = pro_df['title_authors_stems'].apply(lambda x:senti(x))
    pro_df['positive'] = pro_df[pro_df['title_authors_stems_sentiments']>=0]['title_authors_stems_sentiments']
    pro_df['negative'] = pro_df[pro_df['title_authors_stems_sentiments']<0]['title_authors_stems_sentiments']
    pro_df[['positive','negative']] = pro_df[['positive','negative']].fillna(0)
    pro_df['positive'] = MinMaxScaler().fit_transform(pro_df['positive'])
    pro_df['negative'] = MinMaxScaler().fit_transform(abs(pro_df['negative']))

    cmap = matplotlib.cm.get_cmap('Greens')

    def value_color_pos(x):
        cmap = matplotlib.cm.get_cmap('Greens')
        rgba = cmap(x)
        return matplotlib.colors.rgb2hex(rgba)

    def value_color_neg(x):
        cmap = matplotlib.cm.get_cmap('Reds')
        rgba = cmap(x)
        return matplotlib.colors.rgb2hex(rgba)
    pro_df['positive'] = pro_df[pro_df['title_authors_stems_sentiments']>=0]['positive'].apply(lambda x:value_color_pos(x))
    pro_df['negative'] = pro_df[pro_df['title_authors_stems_sentiments']<0]['negative'].apply(lambda x:value_color_neg(x))
    pro_df['col'] = pro_df['positive'].combine_first(pro_df['negative'])
    return pro_df['col']



def dcc_get_senti_colors(pro_df):
    cs = pd.read_csv('Senti_words.csv',index_col=0)
    word_set = set(cs.index.values)
    word_dict = cs.to_dict()['score']

    def senti(x,word_dict=word_dict,word_set=word_set):
        x = str(x).split()
        val =  [word_dict[i] for i in x if i in word_set]
        if val:
            val =  sum(val)/len(val)
        else:
            val =  0
        return val

    pro_df['sentiments'] = pro_df['Abstract'].apply(lambda x:senti(x))
    pro_df['positive'] = pro_df[pro_df['sentiments']>=0]['sentiments']
    pro_df['negative'] = pro_df[pro_df['sentiments']<0]['sentiments']
    pro_df[['positive','negative']] = pro_df[['positive','negative']].fillna(0)
    pro_df['positive'] = MinMaxScaler().fit_transform(np.array(pro_df['positive']).reshape(1,-1)).reshape(pro_df['positive'].shape[0])
    pro_df['negative'] = MinMaxScaler().fit_transform(abs(np.array(pro_df['negative'])).reshape(1,-1)).reshape(pro_df['negative'].shape[0])

    cmap = matplotlib.cm.get_cmap('Greens')

    def value_color_pos(x):
        cmap = matplotlib.cm.get_cmap('Greens')
        rgba = cmap(x)
        return matplotlib.colors.rgb2hex(rgba)

    def value_color_neg(x):
        cmap = matplotlib.cm.get_cmap('Reds')
        rgba = cmap(x)
        return matplotlib.colors.rgb2hex(rgba)
    pro_df['positive'] = pro_df[pro_df['sentiments']>=0]['positive'].apply(lambda x:value_color_pos(x))
    pro_df['negative'] = pro_df[pro_df['sentiments']<0]['negative'].apply(lambda x:value_color_neg(x))
    pro_df['col'] = pro_df['positive'].combine_first(pro_df['negative'])
    return pro_df['col']

def initialize():
    gpro_df = pd.read_csv('Geo/ProFreports.csv',index_col=0,encoding='utf-8')
    gpro_df.reset_index(inplace=True)
    gpro_df = gpro_df.replace(np.nan,'',regex=True)
    gauthors_titles_model = word2vec.Word2Vec.load('Geo/all_titles_authors_model')
    gauthors_titles_vocab = set(gauthors_titles_model.wv.index2word)
    gauthors_titles_normalised = joblib.load('Geo/normalised.pkl')
    gauthors_title_author_mat = joblib.load('Geo/title_author_mat.pkl')
    logging.info("Geo Neighbours Loaded")
    logging.info("Geo Data Frame Loaded")
    logging.info(len(gauthors_titles_vocab))
    gpro_df['col'] = get_senti_colors(gpro_df)
    gword_to_wordID_file = open("Geo/Geo_word_to_wordID_file", "rb")
    gword_to_wordID = pickle.load(gword_to_wordID_file)
    gdocuments_to_concepts_file = open("Geo/Geo_documents_to_concepts_file", "rb")
    gdocuments_to_concepts = pickle.load(gdocuments_to_concepts_file)
    gconcepts_to_words_file = open("Geo/Geo_concepts_to_words_file", "rb")
    gconcepts_to_words = pickle.load(gconcepts_to_words_file)
    gtfwords = joblib.load("Geo/words.pkl")
    gidf = joblib.load("Geo/idf.pkl")
    gnorm = joblib.load("Geo/norm.pkl")

    
    lpro_df = pd.read_csv('Livingston/ProFreports.csv',index_col=0,encoding='utf-8')
    lpro_df.reset_index(inplace=True)
    lpro_df = lpro_df.replace(np.nan,'',regex=True)
    lauthors_titles_model = word2vec.Word2Vec.load('Livingston/all_titles_authors_model')
    lauthors_titles_vocab = set(lauthors_titles_model.wv.index2word)
    lauthors_titles_normalised = joblib.load('Livingston/normalised.pkl')
    lauthors_title_author_mat = joblib.load('Livingston/title_author_mat.pkl')
    logging.info("Livingston Neighbours Loaded")
    logging.info("Livingston Data Frame Loaded")
    logging.info(len(lauthors_titles_vocab))
    lpro_df['col'] = get_senti_colors(lpro_df)
    lword_to_wordID_file = open("Livingston/Livingston_word_to_wordID_file", "rb")
    lword_to_wordID = pickle.load(lword_to_wordID_file)
    ldocuments_to_concepts_file = open("Livingston/Livingston_documents_to_concepts_file", "rb")
    ldocuments_to_concepts = pickle.load(ldocuments_to_concepts_file)
    lconcepts_to_words_file = open("Livingston/Livingston_concepts_to_words_file", "rb")
    lconcepts_to_words = pickle.load(lconcepts_to_words_file)
    ltfwords = joblib.load("Livingston/words.pkl")
    lidf = joblib.load("Livingston/idf.pkl")
    lnorm = joblib.load("Livingston/norm.pkl")

    vpro_df = pd.read_csv('Virgo/ProFreports.csv',index_col=0,encoding='utf-8')
    vpro_df.reset_index(inplace=True)
    vpro_df = vpro_df.replace(np.nan,'',regex=True)
    vauthors_titles_model = word2vec.Word2Vec.load('Virgo/all_titles_authors_model')
    vauthors_titles_vocab = set(vauthors_titles_model.wv.index2word)
    vauthors_titles_normalised = joblib.load('Virgo/normalised.pkl')
    vauthors_title_author_mat = joblib.load('Virgo/title_author_mat.pkl')
    logging.info("Virgo Neighbours Loaded")
    logging.info("Virgo Data Frame Loaded")
    logging.info(len(vauthors_titles_vocab))
    vpro_df['col'] = get_senti_colors(vpro_df)
    vword_to_wordID_file = open("Virgo/Virgo_word_to_wordID_file", "rb")
    vword_to_wordID = pickle.load(vword_to_wordID_file)
    vdocuments_to_concepts_file = open("Virgo/Virgo_documents_to_concepts_file", "rb")
    vdocuments_to_concepts = pickle.load(vdocuments_to_concepts_file)
    vconcepts_to_words_file = open("Virgo/Virgo_concepts_to_words_file", "rb")
    vconcepts_to_words = pickle.load(vconcepts_to_words_file)
    vtfwords = joblib.load("Virgo/words.pkl")
    vidf = joblib.load("Virgo/idf.pkl")
    vnorm = joblib.load("Virgo/norm.pkl")

    hpro_df = pd.read_csv('Hanford/ProFreports.csv',index_col=0,encoding='utf-8')
    hpro_df.reset_index(inplace=True)
    hpro_df = hpro_df.replace(np.nan,'',regex=True)
    hauthors_titles_model = word2vec.Word2Vec.load('Hanford/all_titles_authors_model')
    hauthors_titles_vocab = set(hauthors_titles_model.wv.index2word)
    hauthors_titles_normalised = joblib.load('Hanford/normalised.pkl')
    hauthors_title_author_mat = joblib.load('Hanford/title_author_mat.pkl')
    logging.info("Hanford Neighbours Loaded")
    logging.info("Hanford Data Frame Loaded")
    logging.info(len(hauthors_titles_vocab))
    hpro_df['col'] = get_senti_colors(hpro_df)
    hword_to_wordID_file = open("Hanford/Hanford_word_to_wordID_file", "rb")
    hword_to_wordID = pickle.load(hword_to_wordID_file)
    hdocuments_to_concepts_file = open("Hanford/Hanford_documents_to_concepts_file", "rb")
    hdocuments_to_concepts = pickle.load(hdocuments_to_concepts_file)
    hconcepts_to_words_file = open("Hanford/Hanford_concepts_to_words_file", "rb")
    hconcepts_to_words = pickle.load(hconcepts_to_words_file)
    htfwords = joblib.load("Hanford/words.pkl")
    hidf = joblib.load("Hanford/idf.pkl")
    hnorm = joblib.load("Hanford/norm.pkl")

    model = word2vec.Word2Vec.load("rscr/word2vec_all.model")
    dcc_df = pd.read_csv('DCC/DCC.csv')
    dcctfwords = joblib.load("DCC/dccwords.pkl")
    dccidf = joblib.load("DCC/dccidf.pkl")
    dccnorm = joblib.load("DCC/dccnorm.pkl")
    dcc_df = pd.read_csv('DCC/DCC.csv',encoding='utf-8') 
    dcc_df['col'] = dcc_get_senti_colors(dcc_df)

    try:
        algo = pickle.load(open("rscr/algo", "rb"))
    except:
        algo = 1
        pickle.dump(algo,open("rscr/algo", "wb"))
    return lpro_df ,lauthors_titles_model ,lauthors_titles_vocab ,lauthors_titles_normalised ,lauthors_title_author_mat ,lword_to_wordID ,ldocuments_to_concepts ,lconcepts_to_words ,ltfwords ,lidf ,lnorm ,vpro_df ,vauthors_titles_model ,vauthors_titles_vocab ,vauthors_titles_normalised ,vauthors_title_author_mat ,vword_to_wordID ,vdocuments_to_concepts ,vconcepts_to_words ,vtfwords ,vidf ,vnorm ,hpro_df ,hauthors_titles_model ,hauthors_titles_vocab ,hauthors_titles_normalised ,hauthors_title_author_mat ,hword_to_wordID ,hdocuments_to_concepts ,hconcepts_to_words ,htfwords ,hidf ,hnorm, gpro_df ,gauthors_titles_model ,gauthors_titles_vocab ,gauthors_titles_normalised ,gauthors_title_author_mat ,gword_to_wordID ,gdocuments_to_concepts ,gconcepts_to_words ,gtfwords ,gidf ,gnorm ,model ,dcc_df ,dcctfwords ,dccidf ,dccnorm ,dcc_df ,algo
lpro_df ,lauthors_titles_model ,lauthors_titles_vocab ,lauthors_titles_normalised ,lauthors_title_author_mat ,lword_to_wordID ,ldocuments_to_concepts ,lconcepts_to_words ,ltfwords ,lidf ,lnorm ,vpro_df ,vauthors_titles_model ,vauthors_titles_vocab ,vauthors_titles_normalised ,vauthors_title_author_mat ,vword_to_wordID ,vdocuments_to_concepts ,vconcepts_to_words ,vtfwords ,vidf ,vnorm ,hpro_df ,hauthors_titles_model ,hauthors_titles_vocab ,hauthors_titles_normalised ,hauthors_title_author_mat ,hword_to_wordID ,hdocuments_to_concepts ,hconcepts_to_words ,htfwords ,hidf ,hnorm, gpro_df ,gauthors_titles_model ,gauthors_titles_vocab ,gauthors_titles_normalised ,gauthors_title_author_mat ,gword_to_wordID ,gdocuments_to_concepts ,gconcepts_to_words ,gtfwords ,gidf ,gnorm ,model ,dcc_df ,dcctfwords ,dccidf ,dccnorm ,dcc_df ,algo = initialize()    

@app.route("/sendchat",methods=['POST']) 
def sendchat():
    url = 'https://api.dialogflow.com/v1/query?v=20150910'
    payload = {
        "lang": "en",
        "query": request.form['query'],
        "sessionId": request.form['sessionId'],
    }
    headers = {'content-type': 'application/json', 'Authorization': 'Bearer 4daab51f399b4142832a4d95f674a004'}
    r = requests.post(url, json=payload, headers=headers)
    x = r.json()
    logging.info(x)
    return (x['result']['fulfillment']['messages'][0]['speech'])

@app.route("/robots.txt")
def robots_txt():
    Disallow = lambda string: 'Disallow: {0}'.format(string)
    response = make_response("User-agent: *\n{0}\n".format("\n".join([
        Disallow('/')
    ])))
    response.headers["content-type"] = "text/plain"
    return response 

@app.route('/auto_update')
@requires_auth
def autoupdate():
    subprocess.Popen(['python','autoupdate_han.py'])
    subprocess.Popen(['python','autoupdate_liv.py'])
    subprocess.Popen(['python','autoupdate_vir.py'])
    subprocess.Popen(['python','autoupdate_geo.py'])
    subprocess.Popen(['python','autoupdate_dcc.py'])
    return "Data set has now been updated"
 
@app.route('/refresh_data')
@requires_auth
def refresh():
    global lpro_df ,lauthors_titles_model ,lauthors_titles_vocab ,lauthors_titles_normalised ,lauthors_title_author_mat ,lword_to_wordID ,ldocuments_to_concepts ,lconcepts_to_words ,ltfwords ,lidf ,lnorm ,vpro_df ,vauthors_titles_model ,vauthors_titles_vocab ,vauthors_titles_normalised ,vauthors_title_author_mat ,vword_to_wordID ,vdocuments_to_concepts ,vconcepts_to_words ,vtfwords ,vidf ,vnorm ,hpro_df ,hauthors_titles_model ,hauthors_titles_vocab ,hauthors_titles_normalised ,hauthors_title_author_mat ,hword_to_wordID ,hdocuments_to_concepts ,hconcepts_to_words ,htfwords ,hidf ,hnorm, gpro_df ,gauthors_titles_model ,gauthors_titles_vocab ,gauthors_titles_normalised ,gauthors_title_author_mat ,gword_to_wordID ,gdocuments_to_concepts ,gconcepts_to_words ,gtfwords ,gidf ,gnorm ,model ,dcc_df ,dcctfwords ,dccidf ,dccnorm ,dcc_df ,algo
    lpro_df ,lauthors_titles_model ,lauthors_titles_vocab ,lauthors_titles_normalised ,lauthors_title_author_mat ,lword_to_wordID ,ldocuments_to_concepts ,lconcepts_to_words ,ltfwords ,lidf ,lnorm ,vpro_df ,vauthors_titles_model ,vauthors_titles_vocab ,vauthors_titles_normalised ,vauthors_title_author_mat ,vword_to_wordID ,vdocuments_to_concepts ,vconcepts_to_words ,vtfwords ,vidf ,vnorm ,hpro_df ,hauthors_titles_model ,hauthors_titles_vocab ,hauthors_titles_normalised ,hauthors_title_author_mat ,hword_to_wordID ,hdocuments_to_concepts ,hconcepts_to_words ,htfwords ,hidf ,hnorm, gpro_df ,gauthors_titles_model ,gauthors_titles_vocab ,gauthors_titles_normalised ,gauthors_title_author_mat ,gword_to_wordID ,gdocuments_to_concepts ,gconcepts_to_words ,gtfwords ,gidf ,gnorm ,model ,dcc_df ,dcctfwords ,dccidf ,dccnorm ,dcc_df ,algo = initialize()    
    return "Data set has now been refreshed."
 

@app.route("/")
def landing():
    user_tracking_df = pd.read_csv('user_tracking.csv',index_col=0)
    user = []
    user.append(request.headers.get('User-Agent'))
    user.append(request.remote_addr)
    user.append("/")
    user_tracking_df.loc[user_tracking_df.shape[0]+1] = user
    user_tracking_df.to_csv('user_tracking.csv')
    hitcount = user_tracking_df.shape[0]
    return render_template('landing.html',hitcount=hitcount)

@app.route("/savefeedback",methods=['POST','GET'])
def feedback():
    logging.info("saving feedback")
    feedback_df = pd.read_csv('feedback.csv')
    user = []
    user.append(request.headers.get('User-Agent'))
    user.append(request.remote_addr)
    user.append(request.form['query'])
    feedback_df.loc[feedback_df.shape[0]+1] = user
    feedback_df.to_csv('feedback.csv')
    return "saved"

@app.route("/gauthors_titles")
def gauthor_titles():
        user_tracking_df = pd.read_csv('user_tracking.csv',index_col=0)
        user = []
        user.append(request.headers.get('User-Agent'))
        user.append(request.remote_addr)
        user.append("/gauthors_titles")
        user_tracking_df.loc[user_tracking_df.shape[0]+1] = user 
        user_tracking_df.to_csv('user_tracking.csv')
        hitcount = user_tracking_df.shape[0]
        trend = gpro_df[:50].sort_values(by='comments',ascending=False)[['title','index']]
        trend['index'] = trend['index'].astype('object')
        trend = trend[:5].values.tolist()
        logging.info(trend)
        return render_template('gauthors_titles.html',hitcount=hitcount,trend=trend,sessionId=random.randint(1,99999999999999999999))

@app.route("/vauthors_titles")
def vauthor_titles():
    user_tracking_df = pd.read_csv('user_tracking.csv',index_col=0)
    user = []
    user.append(request.headers.get('User-Agent'))
    user.append(request.remote_addr)
    user.append("/vauthors_titles")
    user_tracking_df.loc[user_tracking_df.shape[0]+1] = user 
    user_tracking_df.to_csv('user_tracking.csv')
    hitcount = user_tracking_df.shape[0]
    trend = vpro_df[:50].sort_values(by='comments',ascending=False)[['title','index']]
    trend['index'] = trend['index'].astype('object')
    trend = trend[:5].values.tolist()
    logging.info(trend)
    return render_template('vauthors_titles.html',hitcount=hitcount,trend=trend,sessionId=random.randint(1,99999999999999999999))

@app.route("/lauthors_titles")
def lauthor_titles():
    user_tracking_df = pd.read_csv('user_tracking.csv',index_col=0)
    user = []
    user.append(request.headers.get('User-Agent'))
    user.append(request.remote_addr)
    user.append("/lauthors_titles")
    user_tracking_df.loc[user_tracking_df.shape[0]+1] = user 
    user_tracking_df.to_csv('user_tracking.csv')
    hitcount = user_tracking_df.shape[0]
    trend = lpro_df[:50].sort_values(by='comments',ascending=False)[['title','index']]
    trend['index'] = trend['index'].astype('object')
    trend = trend[:5].values.tolist()
    logging.info(trend)
    return render_template('lauthors_titles.html',hitcount=hitcount,trend=trend,sessionId=random.randint(1,99999999999999999999))


@app.route("/hauthors_titles")
def hauthor_titles():
    user_tracking_df = pd.read_csv('user_tracking.csv',index_col=0)
    user = []
    user.append(request.headers.get('User-Agent'))
    user.append(request.remote_addr)
    user.append("/hauthors_titles")
    user_tracking_df.loc[user_tracking_df.shape[0]+1] = user 
    user_tracking_df.to_csv('user_tracking.csv')
    hitcount = user_tracking_df.shape[0]
    trend = hpro_df[:50].sort_values(by='comments',ascending=False)[['title','index']]
    trend['index'] = trend['index'].astype('object')
    trend = trend[:5].values.tolist()
    logging.info(trend)
    return render_template('hauthors_titles.html',hitcount=hitcount,trend=trend,sessionId=random.randint(1,99999999999999999999))



@app.route("/changealgo")
def changealgo():
    return render_template('changealgo.html')

@app.route("/changetoalgo", methods = ['POST'])
def changetoalgo():
    global algo
    if request.form['algo'] == "svd":
        algo = 2
    elif request.form['algo'] == "word2vec":
        algo = 0
    else:
        algo = 1
    with open("rscr/algo", 'wb') as writeFile:
        pickle.dump(algo,writeFile)
    return str('changed to '+request.form['algo'])


@app.route("/make_report")
def make_report():
    user_tracking_df = pd.read_csv('user_tracking.csv',index_col=0)
    user = []
    user.append(request.headers.get('User-Agent'))
    user.append(request.remote_addr)
    user.append("/make_report")
    user_tracking_df.loc[user_tracking_df.shape[0]+1] = user 
    user_tracking_df.to_csv('user_tracking.csv')
    hitcount = user_tracking_df.shape[0]
    return render_template('report.html',hitcount=hitcount)

@app.route("/lsearch_authors_titles",methods=['POST','GET'])
def lsearch_authors_titles():
    global algo
    logging.info("Entered Search")
    query = request.form['term']
    logging.info(query)
    query_log_df = pd.read_csv('user_tracking.csv',index_col=0)
    query_row = []
    query_row.append(request.remote_addr)
    query_row.append("Livingston")
    query_row.append(query)
    query_log_df.loc[query_log_df.shape[0]+1] = query_row
    query_log_df.to_csv('user_tracking.csv')
    logging.info(algo)
    if len(query)>=3:
                    logging.info("Searching...")
                    if algo == 0:
                        top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted = get_word2vec_result(query,lauthors_titles_vocab,lauthors_titles_model,lauthors_titles_normalised,lauthors_title_author_mat,lpro_df,False)
                    elif algo == 1:
                        top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted = get_tfidf_word2wiki_result(query,lauthors_titles_normalised,lpro_df,ltfwords,lidf,lnorm,False)
                    elif algo == 2:
                        top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted = get_tfidf_svd_result(query,lauthors_titles_normalised,lpro_df,lword_to_wordID,ldocuments_to_concepts,lconcepts_to_words,False)
                    if(top_searches):
                        results =1 
                    else:
                        results=None
                    logging.info(author_titles)
                    dcc_result, dcc_author_name, dcc_author = get_dcc_result(query)
                    return jsonify(results = results,top_searches=top_searches,top_searches_sorted = top_searches_sorted,names=author_names,author_titles=author_titles,author_titles_sorted=author_titles_sorted,images=images,images_sorted=images_sorted,dcc_result=dcc_result,dcc_author_name=dcc_author_name,dcc_author=dcc_author)                    
    else:    
                    logging.info("Too Short")
                    return jsonify("")


@app.route("/hsearch_authors_titles",methods=['POST','GET'])
def hsearch_authors_titles():
    global algo
    logging.info("Entered Search")
    query = request.form['term']
    logging.info(query)
    query_log_df = pd.read_csv('user_tracking.csv',index_col=0)
    query_row = []
    query_row.append(request.remote_addr)
    query_row.append("Hanford")
    query_row.append(query)
    query_log_df.loc[query_log_df.shape[0]+1] = query_row
    query_log_df.to_csv('user_tracking.csv')
    logging.info(algo)
    if len(query)>=3:
                    logging.info("Searching...")
                    if algo == 0:
                        top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted = get_word2vec_result(query,hauthors_titles_vocab,hauthors_titles_model,hauthors_titles_normalised,hauthors_title_author_mat,hpro_df,False)
                    elif algo == 1:
                        top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted = get_tfidf_word2wiki_result(query,hauthors_titles_normalised,hpro_df,htfwords,hidf,hnorm,False)
                    elif algo == 2:
                        top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted = get_tfidf_svd_result(query,hauthors_titles_normalised,hpro_df,hword_to_wordID,hdocuments_to_concepts,hconcepts_to_words,False)
                    if(top_searches):
                        results =1 
                    else:
                        results=None
                    dcc_result, dcc_author_name, dcc_author = get_dcc_result(query)
                    return jsonify(results = results,top_searches=top_searches,top_searches_sorted = top_searches_sorted,names=author_names,author_titles=author_titles,author_titles_sorted=author_titles_sorted,images=images,images_sorted=images_sorted,dcc_result=dcc_result,dcc_author_name=dcc_author_name,dcc_author=dcc_author)                    
    else:   
                    logging.info("Too Short")
                    return jsonify("")

@app.route("/vsearch_authors_titles",methods=['POST','GET'])
def vsearch_authors_titles():
    global algo
    logging.info("Entered Search")
    query = request.form['term']
    logging.info(query)
    query_log_df = pd.read_csv('user_tracking.csv',index_col=0)
    query_row = []
    query_row.append(request.remote_addr)
    query_row.append("Virgo")
    query_row.append(query)
    query_log_df.loc[query_log_df.shape[0]+1] = query_row
    query_log_df.to_csv('user_tracking.csv')
    logging.info(algo)
    if len(query)>=3:
                    logging.info("Searching...")
                    if algo == 0:
                        top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted = get_word2vec_result(query,vauthors_titles_vocab,vauthors_titles_model,vauthors_titles_normalised,vauthors_title_author_mat,vpro_df,False)
                    elif algo == 1:
                        top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted = get_tfidf_word2wiki_result(query,vauthors_titles_normalised,vpro_df,vtfwords,vidf,vnorm,False)
                    elif algo == 2:
                        top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted = get_tfidf_svd_result(query,vauthors_titles_normalised,vpro_df,vword_to_wordID,vdocuments_to_concepts,vconcepts_to_words,False)
                    if(top_searches):
                        results =1 
                    else:
                        results=None
                    dcc_result, dcc_author_name, dcc_author = get_dcc_result(query)
                    return jsonify(results = results,top_searches=top_searches,top_searches_sorted = top_searches_sorted,names=author_names,author_titles=author_titles,author_titles_sorted=author_titles_sorted,images=images,images_sorted=images_sorted,dcc_result=dcc_result,dcc_author_name=dcc_author_name,dcc_author=dcc_author)                    
    else:   
                    logging.info("Too Short")
                    return jsonify("")


@app.route("/gsearch_authors_titles",methods=['POST','GET'])
def gsearch_authors_titles():
    global algo
    logging.info("Entered Search")
    query = request.form['term']
    logging.info(query)
    query_log_df = pd.read_csv('user_tracking.csv',index_col=0)
    query_row = []
    query_row.append(request.remote_addr)
    query_row.append("GEO")
    query_row.append(query)
    query_log_df.loc[query_log_df.shape[0]+1] = query_row
    query_log_df.to_csv('user_tracking.csv')
    logging.info(algo)
    if algo == 0:
        top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted = get_word2vec_result(query,gauthors_titles_vocab,gauthors_titles_model,gauthors_titles_normalised,gauthors_title_author_mat,gpro_df,True)
    elif algo == 1:
        top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted = get_tfidf_word2wiki_result(query,gauthors_titles_normalised,gpro_df,gtfwords,gidf,gnorm,True)
    elif algo == 2:
        top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted = get_tfidf_svd_result(query,gauthors_titles_normalised,gpro_df,gword_to_wordID,gdocuments_to_concepts,gconcepts_to_words,True)
    if(top_searches):
        results =1 
    else:
        results=None
    dcc_result, dcc_author_name, dcc_author = get_dcc_result(query)
    return jsonify(results = results,top_searches=top_searches,top_searches_sorted = top_searches_sorted,names=author_names,author_titles=author_titles,author_titles_sorted=author_titles_sorted,images=images,images_sorted=images_sorted,dcc_result=dcc_result,dcc_author_name=dcc_author_name,dcc_author=dcc_author)                    
@app.route("/reportl",methods=['POST','GET'])
def reportl():
        logging.info("Entered Search")
        query = request.form['term']
        logging.info(query)
        query_log_df = pd.read_csv('user_tracking.csv',index_col=0)
        query_row = []
        query_row.append(request.remote_addr)
        query_row.append("report")
        query_row.append(query)
        query_log_df.loc[query_log_df.shape[0]+1] = query_row
        query_log_df.to_csv('user_tracking.csv')
        hitcount=str(query_log_df.shape[0])
        if len(query)>=3:
                        clear_files = ['static/img/reports/'+i for i in os.listdir('static/img/reports')]
                        for i in clear_files:
                            os.remove(i)

                        logging.info("Searching...")
                        top_searches_l = get_report(query,lauthors_titles_vocab,lauthors_titles_model,lauthors_titles_normalised,lauthors_title_author_mat,lpro_df,hitcount+'l')
                        logging.info(top_searches_l)

                        if(top_searches_l):
                            results =1 
                        else:
                            results=None
                            
                        return jsonify(results = results,hits=hitcount)

        else:
                            logging.info("Too Short")
                            return jsonify("")

@app.route("/reporth",methods=['POST','GET'])
def reporth():
        logging.info("Entered Search")
        query = request.form['term']
        logging.info(query)
        query_log_df = pd.read_csv('user_tracking.csv',index_col=0)
        hitcount=str(query_log_df.shape[0])
        if len(query)>=3:
                        top_searches_h = get_report(query,hauthors_titles_vocab,hauthors_titles_model,hauthors_titles_normalised,hauthors_title_author_mat,hpro_df,hitcount+'h')
                        logging.info(top_searches_h)

                        if(top_searches_h):
                            results =1 
                        else:
                            results=None
                            
                        return jsonify(results = results,hits=hitcount)

        else:
                            logging.info("Too Short")
                            return jsonify("")
@app.route("/reportv",methods=['POST','GET'])
def reportv():
        logging.info("Entered Search")
        query = request.form['term']
        logging.info(query)
        query_log_df = pd.read_csv('user_tracking.csv',index_col=0)
        hitcount=str(query_log_df.shape[0])
        if len(query)>=3:
                        top_searches_v = get_report(query,vauthors_titles_vocab,vauthors_titles_model,vauthors_titles_normalised,vauthors_title_author_mat,vpro_df,hitcount+'v')
                        top_searches_merged = get_merged_report(query,hitcount)
                        logging.info(top_searches_v)
                        logging.info(top_searches_merged)

                        if(top_searches_v and top_searches_merged):
                            results =1 
                        else:
                            results=None
                            
                        return jsonify(results = results,hits=hitcount)

        else:
                            logging.info("Too Short")
                            return jsonify("")
@app.route("/alerts")
def alerts():
    user_tracking_df = pd.read_csv('user_tracking.csv',index_col=0)
    user = []
    user.append(request.headers.get('User-Agent'))
    user.append(request.remote_addr)
    user.append("/alerts")
    user_tracking_df.loc[user_tracking_df.shape[0]+1] = user 
    user_tracking_df.to_csv('user_tracking.csv')
    hitcount = user_tracking_df.shape[0]
    trend = vpro_df[:50].sort_values(by='comments',ascending=False)[['title','index']]
    trend['index'] = trend['index'].astype('object')
    trend = trend[:5].values.tolist()
    logging.info(trend)
    return render_template('alerts.html',hitcount=hitcount)

@app.route('/alerts_sub', methods=['POST'])
def alerts_sub():
    email = request.form['email']
    keywords = request.form['kw']
    obs = request.form['obs']
    alerts_user = pd.read_csv('alerts_user.csv',index_col=0)
    alerts_user.loc[alerts_user.shape[0]+1] = [email,keywords,obs]
    alerts_user.drop_duplicates(subset=['email'],keep='last')
    alerts_user.to_csv('alerts_user.csv')
    alerts_email()
    return render_template('alert_sub.html') #"Success"

@app.route("/search_authors_titles_contents",methods=['POST','GET'])
def search_authors_titles_contents():
        logging.info("Entered Search")
        logging.info("bla")
        query = request.form['term']
        logging.info(query)
        if len(query)>=10:
                        logging.info("Searching...")
                        top_searches,author_names,author_titles = get_word2vec_result(query,authors_titles_contents_vocab,authors_titles_contents_model,authors_titles_contents_normalised,pro_df)
                        logging.info(top_searches)
                        logging.info(author_titles)
                        if(top_searches):
                            results =1 
                        else:
                            results=None
                        return jsonify(results = results,top_searches=top_searches,names=author_names,author_titles=author_titles)
        else:    
                        logging.info("Too Short")
                        return jsonify("")
    
@app.route("/dashboard",methods=['POST','GET'])
def dashboard():
    return render_template('dashboard.html')

@app.route("/authors_titles_contents")
def author_titles_contents():
    return render_template('authors_titles_contents.html')


#-----------------------NLP Funtions ---------------------

def knn(normalised,title_author_mat,feature_vec):
    queryvec = (feature_vec*feature_vec).sum()
    distances = np.matmul(title_author_mat,feature_vec)/(normalised*queryvec)
    return list(distances),list(np.argsort(distances)[::-1])

def tokstopandstem(text):
    #remove punctuation and split into seperate words
    words = re.findall(r'[0-9]*[a-zA-Z]+[0-9]*[a-zA-Z]*', text.lower(),flags = re.UNICODE) 
    #This is the simple way to remove stop words
    important_words=[]
    for word in words:
        if word not in stopwords.words('english'):
            important_words.append(word)
    stems = [stemmer.stem(t) for t in important_words]
    return " ".join(stems)

def get_feature_vecs(text,model,vocab,num_features):
    text = text.split()
    feature_vec = np.zeros((num_features,),dtype='float32')
    nwords = 0
    for word in text:
        if word in vocab:
            nwords=nwords+1
            #logging.info(word)
            #logging.info(model[word])
            feature_vec = np.add(feature_vec,model[word])
    feature_vec = np.divide(feature_vec,nwords)
    return feature_vec 

def search_cont(dfrm, query):
    # Improving query search: Addition by Sheelu & Nikhil
    xxxx = pd.Series(dfrm['content'])
    cccc = pd.Series(dfrm['title'])
    dfm = dfrm
    qk = query.split(' ')
    yyy = []
    ttt = []
    cntt = 0
    
    for items in qk:
        yyy.append(list([xxxx.str.contains(items,case=False)]))
        ttt.append(list([cccc.str.contains(items,case=False)]))
        if cntt == 0:
            dft = np.ones(len(yyy), dtype=bool)
        #tmp = np.array(dft & np.array(yyy[cntt][0], dtype=bool))
        tmp = np.array(dft & np.array(yyy[cntt][0], dtype=bool)) | (dft & np.array(ttt[cntt][0], dtype=bool))
        #tmp = np.array((dft & np.array(ttt[cntt][0], dtype=bool)))
        if tmp.any():
            dft = tmp
        else:
            dft = dft
            cntt-=1
        cntt+=1
        dfm = dfrm[dft]
    return(dfm)

def cosine_similarity(a, b):
    return np.dot(a,b) / (math.sqrt(np.dot(a, a)) * math.sqrt(np.dot(b, b)))

def plot_comb(x,label, flag=2):
    xx = []
    cnt =[]
    for i in range(len(x)):
        p = x.index[i][0]+((x.index[i][1]-1)/12)
        xx.append(p)
        cnt.append(x['index'][i])
    if flag == 0:
        plt.gca().set_axis_bgcolor("lightslategray")
        plt.bar(xx, cnt, label=label, width=0.05, linewidth=2, edgecolor='w', facecolor='#9999ff', alpha=0.99)#, linestyle='-')
    elif flag == 1:
        plt.bar(xx, cnt, label=label, width=0.05, linewidth=2, edgecolor='w', facecolor='#ff9999', alpha=0.7)#, linestyle='--')
    elif flag == 2:
        plt.bar(xx, cnt, label=label, width=0.05, linewidth=2, edgecolor='w', facecolor='#99ff99', alpha=0.55)#, linestyle=':')
    elif flag == 3:
        plt.bar(xx, cnt, label=label, width=0.05, facecolor='r', linewidth=2)
        plt.xlim(np.floor(min(xx)), np.ceil(max(xx)))
    else:
        print ("Wrong Flag")
    plt.xlabel("Year", fontsize=18)
    plt.ylabel("No of posts", fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.grid(True)

def get_report(query,vocab,model,normalised,title_author_mat,pro_df,obs):
    query = tokstopandstem(query)

    #logging.info(query)
    feature_vec = get_feature_vecs(str(query),model,vocab,num_feat)
    #logging.info(list(feature_vec))
    if not np.argwhere(np.isnan(feature_vec)).shape[0]:
        distances, indices =  knn(normalised,title_author_mat,feature_vec)
        search_df = pro_df.iloc[indices]
        search_df.reset_index(inplace=True,drop=True)
        search_df['report_time'] = pd.to_datetime(search_df['report_time'])
        search_df = search_df[search_df['report_time']>dt.date(2010,1,1)]
        search_df = search_cont(search_df, query)
        fig = plt.figure(figsize=(5,5))
        plt.title("Rate of Occurence of %s"%query, fontsize=18)
        plt.grid(True)
        #if search_df.shape[0] > 0:
        xx = pd.DataFrame(search_df['index'].groupby([search_df.report_time.dt.year, search_df.report_time.dt.month]).agg('count'))#.plot()
        plot_comb(x=xx, label='', flag=3)
             
        fig.autofmt_xdate()
        plt.savefig('static/img/reports/timeline_%s.png'%obs)
        plt.figure(figsize=(5,5))
        plt.title("Distribution of sections for %s "%query, fontsize=18)
        search_df['section'] = search_df['section'].apply(lambda x:x[:6])
        #xsec = search_df['section'].value_counts()[search_df['section'].value_counts()>1]
        cs=cm.Set2(np.arange(8)/8.)
        search_df['section'].value_counts().plot(kind='pie', colors=cs, fontsize=11)
        #if xsec.shape[0] > 0:
        #    xsec.plot(kind='pie')
        plt.ylabel("")
        plt.grid(False)
        plt.savefig('static/img/reports/pie_%s.png'%obs)
        logging.info("Single plot %s Complete"%obs)
        #else:
        #    return None
        return 1
    else:
        return None

def get_merged_report(query,hitcount):
    query = tokstopandstem(query)
    #logging.info(query)
    #logging.info(list(feature_vec))
    
    try:
            plt.figure(figsize=(20,5))
            plt.grid(False)
            plt.title("Rate of Occurence of %s"%query, fontsize=18)
            
            feature_vec = get_feature_vecs(str(query),lauthors_titles_model,lauthors_titles_vocab,num_feat)
            distances, indices =  knn(lauthors_titles_normalised,lauthors_title_author_mat,feature_vec)
            search_df = lpro_df.iloc[indices]
            #search_df = search_df.iloc[0:400]
            search_df.reset_index(inplace=True,drop=True)
            search_df['report_time'] = pd.to_datetime(search_df['report_time'])
            search_df = search_df[(search_df['report_time']>dt.date(2010,10,1))]
            search_df = search_cont(search_df, query)
            xl = pd.DataFrame(search_df['index'].groupby([search_df.report_time.dt.year, search_df.report_time.dt.month]).agg('count'))#.plot(label='LLO')
            plot_comb(x=xl, label='LLO', flag=0)

            feature_vec = get_feature_vecs(str(query),hauthors_titles_model,hauthors_titles_vocab,num_feat)
            distances, indices =  knn(hauthors_titles_normalised,hauthors_title_author_mat,feature_vec)
            search_df = hpro_df.iloc[indices]
            #search_df = search_df.iloc[0:400]
            search_df.reset_index(inplace=True,drop=True)
            search_df['report_time'] = pd.to_datetime(search_df['report_time'])
            search_df = search_df[(search_df['report_time']>dt.date(2010,10,1))]
            search_df = search_cont(search_df, query)
            xh = pd.DataFrame(search_df['index'].groupby([search_df.report_time.dt.year, search_df.report_time.dt.month]).agg('count'))#.plot(label='LHO')
            plot_comb(x=xh, label='LHO', flag=1)
            
            feature_vec = get_feature_vecs(str(query),vauthors_titles_model,vauthors_titles_vocab,num_feat)
            distances, indices =  knn(vauthors_titles_normalised,vauthors_title_author_mat,feature_vec)
            search_df = vpro_df.iloc[indices]
            #search_df = search_df.iloc[0:400]
            search_df.reset_index(inplace=True,drop=True)
            search_df['report_time'] = pd.to_datetime(search_df['report_time'])
            search_df = search_df[(search_df['report_time']>dt.date(2010,10,1))]
            search_df = search_cont(search_df, query)
            xv = pd.DataFrame(search_df['index'].groupby([search_df.report_time.dt.year, search_df.report_time.dt.month]).agg('count'))#.plot(label='VIRGO')
            plot_comb(x=xv, label='Virgo', flag=2)
                        
            #plt.legend(loc='upper left')
            #plt.xlabel('Report_time:(year,month)')
            plt.savefig('static/img/reports/timeline%s.png'%hitcount)
            logging.info("Merge plot Complete")
            return 1

    except Exception as err:
        logging.info(err)
        return None

def get_word2vec_result(query,vocab,model,normalised,title_author_mat,pro_df,isgeo):
    logging.info("##############Word2Vec###############")

    query = tokstopandstem(query)
    #logging.info(query)
    feature_vec = get_feature_vecs(str(query),model,vocab,num_feat)
    #logging.info(list(feature_vec))
    if not np.argwhere(np.isnan(feature_vec)).shape[0]:
        distances, indices =  knn(normalised,title_author_mat,feature_vec)
        search_df = pro_df.iloc[indices]
        search_df['report_time'] = pd.to_datetime(search_df['report_time'])
        search_df = search_cont(search_df, query)
        figure = plt.figure()
        plt.title("Rate of Occurence of %s"%query, fontsize=18)
        


        #generating copy of the results for later usage 
        search_df_copy = search_df.copy()

        #Selecting top 30 posts to calculate author ranks 
        search_df = search_df.iloc[:30]

        #Resetting index for calculating author ranks 
        search_df.reset_index(inplace=True,drop=True)

        #Generating ranks for top authors 
        ranks = {}
        for i,j in search_df.groupby('author_id').groups.items():
            j = [1/(index+1) for index in j]
            ranks[i] = sum(j)
        ranks = sorted(ranks.items(), key=lambda x:x[1],reverse=True)
        auths= [auth[0] for auth in ranks][:4]

        #Top authors stored in auths 
        logging.info(auths)


        #Converting the string fomrat of date to python date time format
        search_df['report_time'] = pd.to_datetime(search_df['report_time'])
        search_df_copy['report_time'] = pd.to_datetime(search_df_copy['report_time'])

        #Generating relevent postds sorted by time 
        search_df_sorted = search_df.sort_values(by=['report_time'],ascending=False)
        search_df_sorted['report_time'] = pd.to_datetime(search_df_sorted['report_time'])
        
        top_searches = []
        top_searches_sorted = []
        #Storing author names to be passed
        author_names = auths
        author_1 = []
        author_2 = []
        author_3 = []
        author_4 = []

        def extract_image_urls(img):
            images = []
            for i in img:
                if not i[0]=='':
                    for url in i[0].split(","):
                        images.append([url,i[1],i[2]])
            return images
        #Putting top 30 posts sorted by relevence and time in a list for passing to ajax object
        if isgeo == False:
            top_searches = search_df[['index','title','author_id','report_time','col','code_flag']].values.tolist()
            top_searches_sorted = search_df_sorted[['index','title','author_id','report_time','col','code_flag']].values.tolist()
        else:
            top_searches = search_df[['index','title','author_id','report_time','col','code_flag','content']].values.tolist()
            top_searches_sorted = search_df_sorted[['index','title','author_id','report_time','col','code_flag','content']].values.tolist()

        images = extract_image_urls(search_df_copy[['file_urls','index','title']].values.tolist())
        images_sorted = extract_image_urls(search_df_sorted[['file_urls','index','title']].values.tolist())

        #Authors Sorted according to Report time
        #Getting all the posts of top authors in auths 
        authors = search_df_copy[search_df_copy['author_id'].isin(auths)].head(1000).reset_index(drop=True)

        authors['report_time'] =  pd.to_datetime(authors['report_time'])


        #1) Authors sorted according to time 

        if isgeo == False:
            try:
                author_1_sorted = authors[authors['author_id']==auths[0]][['index','title','author_id','report_time','col','code_flag']]
                author_1_sorted = author_1_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
            except:
                author_1_sorted = None   
            try:
                author_2_sorted = authors[authors['author_id']==auths[1]][['index','title','author_id','report_time','col','code_flag']]
                author_2_sorted = author_2_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
            except:
                author_2_sorted = None 
            try:
                author_3_sorted = authors[authors['author_id']==auths[2]][['index','title','author_id','report_time','col','code_flag']]
                author_3_sorted = author_3_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
            except:
                author_3_sorted = None
            try:
                author_4_sorted = authors[authors['author_id']==auths[3]][['index','title','author_id','report_time','col','code_flag']]
                author_4_sorted = author_4_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
            except:
                author_4_sorted = None      
        
            #2) Authors according to  relevance 
            try:
                author_1 = authors[authors['author_id']==auths[0]][['index','title','author_id','report_time','col','code_flag']].head(number_of_results).values.tolist()
            except:
                author_2 = None
            try:
                author_2 = authors[authors['author_id']==auths[1]][['index','title','author_id','report_time','col','code_flag']].head(number_of_results).values.tolist()
            except:
                author_2 = None
            try:
                author_3 = authors[authors['author_id']==auths[2]][['index','title','author_id','report_time','col','code_flag']].head(number_of_results).values.tolist()
            except:
                author_3 = None
            try:
                author_4 = authors[authors['author_id']==auths[3]][['index','title','author_id','report_time','col','code_flag']].head(number_of_results).values.tolist()
            except:
                author_4 = None
        else:
            try:
                author_1_sorted = authors[authors['author_id']==auths[0]][['index','title','author_id','report_time','col','code_flag','content']]
                author_1_sorted = author_1_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
            except:
                author_1_sorted = None   
            try:
                author_2_sorted = authors[authors['author_id']==auths[1]][['index','title','author_id','report_time','col','code_flag','content']]
                author_2_sorted = author_2_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
            except:
                author_2_sorted = None 
            try:
                author_3_sorted = authors[authors['author_id']==auths[2]][['index','title','author_id','report_time','col','code_flag','content']]
                author_3_sorted = author_3_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
            except:
                author_3_sorted = None
            try:
                author_4_sorted = authors[authors['author_id']==auths[3]][['index','title','author_id','report_time','col','code_flag','content']]
                author_4_sorted = author_4_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
            except:
                author_4_sorted = None      
            
            #2) Authors according to  relevance 
            try:
                author_1 = authors[authors['author_id']==auths[0]][['index','title','author_id','report_time','col','code_flag','content']].head(number_of_results).values.tolist()
            except:
                author_2 = None
            try:
                author_2 = authors[authors['author_id']==auths[1]][['index','title','author_id','report_time','col','code_flag','content']].head(number_of_results).values.tolist()
            except:
                author_2 = None
            try:
                author_3 = authors[authors['author_id']==auths[2]][['index','title','author_id','report_time','col','code_flag','content']].head(number_of_results).values.tolist()
            except:
                author_3 = None
            try:
                author_4 = authors[authors['author_id']==auths[3]][['index','title','author_id','report_time','col','code_flag','content']].head(number_of_results).values.tolist()
            except:
                author_4 = None
        #stacking all authors to a list to pass as ajax response object 
        author_titles = [author_1,author_2,author_3,author_4]
        author_titles_sorted = [author_1_sorted,author_2_sorted,author_3_sorted,author_4_sorted]
        logging.info(top_searches)
        return top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted

def searchQuery(queryWords,tfwords,idf,norm):
    score = {}
    for word in queryWords:
        if word in tfwords:
            print(word)
            for document in tfwords[word]:
                if document not in score:
                    score[document]=0
                score[document] += tfwords[word][document]*math.log10(1+queryWords[word])
    for doc in score:
        score[doc] /= norm[doc]
    sortedScore = {}
    for doc in score:
        sortedScore [score[doc]]=doc
    sortedScore = sorted(sortedScore.items(), reverse=True)
    result = []
    similarity = []
    for doc in sortedScore:
        result.append(doc[1])
        similarity.append(doc[0])
    return result,similarity

def dccSearchQuery(queryWords):
    score = {}
    for word in queryWords:
        if word in dcctfwords:
            for document in dcctfwords[word]:
                if document not in score:
                    score[document]=0
                score[document] +=  dcctfwords[word][document]*math.log10(1+queryWords[word])
    for doc in score:
        score[doc] /= dccnorm[doc]
    sortedScore = {}
    for doc in score:
        sortedScore [score[doc]]=doc
    sortedScore = sorted(sortedScore.items(), reverse=True)
    result = []
    similarity = []
    for doc in sortedScore:
        result.append(doc[1])
        similarity.append(doc[0])
    return result,similarity

def query_func(query,concepts_to_words,word_to_wordID):
    query = tokstopandstem(query)
    query = query.split(" ")
    query_inverted_index = {}

    for word in query:
        if word not in query_inverted_index:
            query_inverted_index[word] = 1
        else:
            query_inverted_index[word] += 1

    query_to_words = np.zeros(shape = (len(word_to_wordID)))
    known_word_found_flag = 0

    for word in query_inverted_index:
        if word in word_to_wordID:
            known_word_found_flag = 1
            query_inverted_index[word] = math.log(1+query_inverted_index[word])
            query_to_words[word_to_wordID[word]] = query_inverted_index[word]

    if known_word_found_flag == 0:
        print("No document matches your query")
        return np.zeros((len(word_to_wordID), (concepts_to_words.shape)[0])), 0

    query_to_concepts = np.dot(query_to_words, concepts_to_words.T)

    return query_to_concepts, 1

def queryPreProcess(text):
    allWords = re.findall(r'[0-9]*[a-zA-Z]+[0-9]*[a-zA-Z]*', text.lower(),flags = re.UNICODE) 
    #This is the simple way to remove stop words
    importantWords=[]
    queryWords = {}
    for word in allWords:
        if word not in stopwords.words('english'):
            importantWords.append(word)
    stems = [stemmer.stem(t) for t in importantWords]
    for word in stems:
        if word not in queryWords:
            queryWords[word] = 0
        queryWords[word] += 1
    newword ={}
    for word in queryWords:
        if model.wv.__contains__(word):
            for t in model.wv.most_similar(word):
                if t[1] < 0.6:
                     break  
                if t[0] in queryWords:
                    continue
            if t[0] in newword:
                if newword[t[0]] < queryWords[word]*t[1]:
                    newword[t[0]] = queryWords[word]*t[1]
            else:
                newword[t[0]] = queryWords[word]*t[1]
    print(newword)
    for word in newword:
        queryWords[word] = newword[word]
    return queryWords

def get_dcc_result(query):
    logging.info("##############DCC##############")
    indices, similarity = dccSearchQuery(queryPreProcess(query))
    search_df = dcc_df.iloc[indices]
    top_searches = search_df[['Title','Author','link','Ligo-Number','col']].head(number_of_results).values.tolist()
    ranks = {}
    i = 0
    for j in top_searches:
        if j[1] not in ranks:
            ranks[j[1]] = 0
        ranks[j[1]] += similarity[i]
        i += 1 
    ranks = sorted(ranks.items(), key=lambda x:x[1],reverse=True)
    auths= [auth[0] for auth in ranks][:4]

    author_names = auths
    author_1 = []
    author_2 = []
    author_3 = []
    author_4 = []
    
    authors = search_df[search_df['Author'].isin(auths)].head(1000)

    try:
        author_1 = authors[authors['Author']==auths[0]][['Title','Author','link','Ligo-Number','col']].head(number_of_results).values.tolist()
    except:
        author_1 = None
    try:
        author_2 = authors[authors['Author']==auths[1]][['Title','Author','link','Ligo-Number','col']].head(number_of_results).values.tolist()
    except:
        author_2 = None
    try:
        author_3 = authors[authors['Author']==auths[2]][['Title','Author','link','Ligo-Number','col']].head(number_of_results).values.tolist()
    except:
        author_3 = None
    try:
        author_4 = authors[authors['Author']==auths[3]][['Title','Author','link','Ligo-Number','col']].head(number_of_results).values.tolist()
    except:
        author_4 = None

    author_titles = [author_1,author_2,author_3,author_4]
    logging.info(top_searches)
    return top_searches,auths,author_titles

def get_tfidf_word2wiki_result(query,normalised,pro_df,tfwords,idf,norm,isgeo):
    logging.info("##############WIKI###############")
    query = tokstopandstem(query)
    tfquery = queryPreProcess(query)
    indices,similarity = searchQuery(tfquery,tfwords,idf,norm)
    if len(indices) == 0:
        return None,None,None,None,None,None,None
    search_df = pro_df.iloc[indices]
    search_df['report_time'] = pd.to_datetime(search_df['report_time'])
    #search_df = search_cont(search_df, query) 
    figure = plt.figure()
    plt.title("Rate of Occurence of %s"%query, fontsize=18)

    #generating copy of the results for later usage 
    search_df_copy = search_df.copy()

    #Selecting top numbers posts to calculate author ranks 
    search_df = search_df.iloc[:number_of_results]

    #Resetting index for calculating author ranks 
    search_df.reset_index(inplace=True,drop=True)

    #Generating ranks for top authors 
    ranks = {}
    i = 0
    for j in search_df[['author_id']].values.tolist():
        if j[0] not in ranks:
            ranks[j[0]] = 0
        ranks[j[0]] += similarity[i]
        i += 1 
    ranks = sorted(ranks.items(), key=lambda x:x[1],reverse=True)
    auths= [auth[0] for auth in ranks][:4]
   
    #Top authors stored in auths 
    logging.info(auths)


    #Converting the string fomrat of date to python date time format
    search_df['report_time'] = pd.to_datetime(search_df['report_time'])
    search_df_copy['report_time'] = pd.to_datetime(search_df_copy['report_time'])

    #Generating relevent postds sorted by time 
    search_df_sorted = search_df.sort_values(by=['report_time'],ascending=False)
    search_df_sorted['report_time'] = pd.to_datetime(search_df_sorted['report_time'])
    
    top_searches = []
    top_searches_sorted = []
    #Storing author names to be passed
    author_names = auths
    author_1 = []
    author_2 = []
    author_3 = []
    author_4 = []


    def extract_image_urls(img):
        images = []
        j = 0
        for i in img:
            if j >= 30:
                break
            if i[0] != '':
                for url in i[0].split(","):
                    images.append([url,i[1],i[2]])
                    j += 1
        return images

    #Putting top 30 posts sorted by relevence and time in a list for passing to ajax object
    if isgeo == False:
        top_searches = search_df[['index','title','author_id','report_time','col','code_flag']].values.tolist()
        top_searches_sorted = search_df_sorted[['index','title','author_id','report_time','col','code_flag']].values.tolist()
    else:
        top_searches = search_df[['index','title','author_id','report_time','col','code_flag','content']].values.tolist()
        top_searches_sorted = search_df_sorted[['index','title','author_id','report_time','col','code_flag','content']].values.tolist()

    images = extract_image_urls(search_df_copy[['file_urls','index','title']].values.tolist())
    images_sorted = extract_image_urls(search_df_sorted[['file_urls','index','title']].values.tolist())

    #Authors Sorted according to Report time
    #Getting all the posts of top authors in auths 
    authors = search_df_copy[search_df_copy['author_id'].isin(auths)].head(1000).reset_index(drop=True)

    authors['report_time'] =  pd.to_datetime(authors['report_time'])


    #1) Authors sorted according to time 

    if isgeo == False:
        try:
            author_1_sorted = authors[authors['author_id']==auths[0]][['index','title','author_id','report_time','col','code_flag']]
            author_1_sorted = author_1_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_1_sorted = None   
        try:
            author_2_sorted = authors[authors['author_id']==auths[1]][['index','title','author_id','report_time','col','code_flag']]
            author_2_sorted = author_2_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_2_sorted = None 
        try:
            author_3_sorted = authors[authors['author_id']==auths[2]][['index','title','author_id','report_time','col','code_flag']]
            author_3_sorted = author_3_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_3_sorted = None
        try:
            author_4_sorted = authors[authors['author_id']==auths[3]][['index','title','author_id','report_time','col','code_flag']]
            author_4_sorted = author_4_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_4_sorted = None      
        
        #2) Authors according to  relevance 
        try:
            author_1 = authors[authors['author_id']==auths[0]][['index','title','author_id','report_time','col','code_flag']].head(number_of_results).values.tolist()
        except:
            author_2 = None
        try:
            author_2 = authors[authors['author_id']==auths[1]][['index','title','author_id','report_time','col','code_flag']].head(number_of_results).values.tolist()
        except:
            author_2 = None
        try:
            author_3 = authors[authors['author_id']==auths[2]][['index','title','author_id','report_time','col','code_flag']].head(number_of_results).values.tolist()
        except:
            author_3 = None
        try:
            author_4 = authors[authors['author_id']==auths[3]][['index','title','author_id','report_time','col','code_flag']].head(number_of_results).values.tolist()
        except:
            author_4 = None
    else:
        try:
            author_1_sorted = authors[authors['author_id']==auths[0]][['index','title','author_id','report_time','col','code_flag','content']]
            author_1_sorted = author_1_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_1_sorted = None   
        try:
            author_2_sorted = authors[authors['author_id']==auths[1]][['index','title','author_id','report_time','col','code_flag','content']]
            author_2_sorted = author_2_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_2_sorted = None 
        try:
            author_3_sorted = authors[authors['author_id']==auths[2]][['index','title','author_id','report_time','col','code_flag','content']]
            author_3_sorted = author_3_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_3_sorted = None
        try:
            author_4_sorted = authors[authors['author_id']==auths[3]][['index','title','author_id','report_time','col','code_flag','content']]
            author_4_sorted = author_4_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_4_sorted = None      
        
        #2) Authors according to  relevance 
        try:
            author_1 = authors[authors['author_id']==auths[0]][['index','title','author_id','report_time','col','code_flag','content']].head(number_of_results).values.tolist()
        except:
            author_2 = None
        try:
            author_2 = authors[authors['author_id']==auths[1]][['index','title','author_id','report_time','col','code_flag','content']].head(number_of_results).values.tolist()
        except:
            author_2 = None
        try:
            author_3 = authors[authors['author_id']==auths[2]][['index','title','author_id','report_time','col','code_flag','content']].head(number_of_results).values.tolist()
        except:
            author_3 = None
        try:
            author_4 = authors[authors['author_id']==auths[3]][['index','title','author_id','report_time','col','code_flag','content']].head(number_of_results).values.tolist()
        except:
            author_4 = None
    #stacking all authors to a list to pass as ajax response object 
    author_titles = [author_1,author_2,author_3,author_4]
    author_titles_sorted = [author_1_sorted,author_2_sorted,author_3_sorted,author_4_sorted]
    logging.info(top_searches)
    return top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted

def get_tfidf_svd_result(query,normalised,pro_df,word_to_wordID,documents_to_concepts,concepts_to_words,isgeo):
    logging.info("############SVD##############")
    query = tokstopandstem(query)
    query_to_concepts, flag = query_func(str(query),concepts_to_words,word_to_wordID)
    if flag == 0:
        print("@@@@@")
        return None,None,None,None,None,None,None
    score = {}
    indices = []
    
    for doc, i in zip(documents_to_concepts, range(len(documents_to_concepts))):
        score[i] = cosine_similarity(np.array(doc), query_to_concepts)
    
    score = sorted(score.items(), key = operator.itemgetter(1), reverse = True)
    score_list = []
    j = 0
    for s, i in zip(score, range(len(score))):
        if i < number_of_results:
            score_list.append(s[1])
        if j < 1001:
            indices.append(s[0])
            j += 1
    if len(indices) == 0:
        return None,None,None,None,None,None,None
    search_df = pro_df.loc[indices]
    search_df['report_time'] = pd.to_datetime(search_df['report_time'])
    figure = plt.figure()

    plt.title("Rate of Occurence of %s"%query, fontsize=18)

    #generating copy of the results for later usage
    search_df_copy = search_df.copy()

    #Selecting top 30 posts to calculate author ranks 
    search_df = search_df.iloc[:number_of_results]

    #Resetting index for calculating author ranks
    ranks = {}
    i = 0
    for j in search_df[['author_id']].values.tolist():
        if j[0] not in ranks:
            ranks[j[0]] = 0
        ranks[j[0]] += score_list[i]
        i += 1

    ranks = sorted(ranks.items(), key=lambda x:x[1],reverse=True)
    auths= [auth[0] for auth in ranks][:4]

    #Top authors stored in auths 
    logging.info(auths)

    #Converting the string fomrat of date to python date time format
    search_df['report_time'] = pd.to_datetime(search_df['report_time'])
    search_df_copy['report_time'] = pd.to_datetime(search_df_copy['report_time'])

    #Generating relevent postds sorted by time 
    search_df_sorted = search_df.sort_values(by=['report_time'],ascending=False)
    search_df_sorted['report_time'] = pd.to_datetime(search_df_sorted['report_time'])
    
    top_searches = []
    top_searches_sorted = []
    #Storing author names to be passed
    author_names = auths
    author_1 = []
    author_2 = []
    author_3 = []
    author_4 = []


    def extract_image_urls(img):
        images = []
        j = 0
        for i in img:
            if j >= 30:
                break
            if i[0] != '':
                for url in i[0].split(","):
                    images.append([url,i[1],i[2]])
                    j += 1
        return images

    #Putting top 30 posts sorted by relevence and time in a list for passing to ajax object 
    if isgeo == False:
        top_searches = search_df[['index','title','author_id','report_time','col','code_flag']].values.tolist()
        top_searches_sorted = search_df_sorted[['index','title','author_id','report_time','col','code_flag']].values.tolist()
    else:
        top_searches = search_df[['index','title','author_id','report_time','col','code_flag','content']].values.tolist()
        top_searches_sorted = search_df_sorted[['index','title','author_id','report_time','col','code_flag','content']].values.tolist()

    images = extract_image_urls(search_df_copy[['file_urls','index','title']].values.tolist())
    images_sorted = extract_image_urls(search_df_sorted[['file_urls','index','title']].values.tolist())

    #Authors Sorted according to Report time
    #Getting all the posts of top authors in auths 
    authors = search_df_copy[search_df_copy['author_id'].isin(auths)].head(1000).reset_index(drop=True)

    authors['report_time'] =  pd.to_datetime(authors['report_time'])


    #1) Authors sorted according to time 

    if isgeo == False:
        try:
            author_1_sorted = authors[authors['author_id']==auths[0]][['index','title','author_id','report_time','col','code_flag']]
            author_1_sorted = author_1_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_1_sorted = None   
        try:
            author_2_sorted = authors[authors['author_id']==auths[1]][['index','title','author_id','report_time','col','code_flag']]
            author_2_sorted = author_2_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_2_sorted = None 
        try:
            author_3_sorted = authors[authors['author_id']==auths[2]][['index','title','author_id','report_time','col','code_flag']]
            author_3_sorted = author_3_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_3_sorted = None
        try:
            author_4_sorted = authors[authors['author_id']==auths[3]][['index','title','author_id','report_time','col','code_flag']]
            author_4_sorted = author_4_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_4_sorted = None      
        
        #2) Authors according to  relevance 
        try:
            author_1 = authors[authors['author_id']==auths[0]][['index','title','author_id','report_time','col','code_flag']].head(number_of_results).values.tolist()
        except:
            author_2 = None
        try:
            author_2 = authors[authors['author_id']==auths[1]][['index','title','author_id','report_time','col','code_flag']].head(number_of_results).values.tolist()
        except:
            author_2 = None
        try:
            author_3 = authors[authors['author_id']==auths[2]][['index','title','author_id','report_time','col','code_flag']].head(number_of_results).values.tolist()
        except:
            author_3 = None
        try:
            author_4 = authors[authors['author_id']==auths[3]][['index','title','author_id','report_time','col','code_flag']].head(number_of_results).values.tolist()
        except:
            author_4 = None
    else:
        try:
            author_1_sorted = authors[authors['author_id']==auths[0]][['index','title','author_id','report_time','col','code_flag','content']]
            author_1_sorted = author_1_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_1_sorted = None   
        try:
            author_2_sorted = authors[authors['author_id']==auths[1]][['index','title','author_id','report_time','col','code_flag','content']]
            author_2_sorted = author_2_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_2_sorted = None 
        try:
            author_3_sorted = authors[authors['author_id']==auths[2]][['index','title','author_id','report_time','col','code_flag','content']]
            author_3_sorted = author_3_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_3_sorted = None
        try:
            author_4_sorted = authors[authors['author_id']==auths[3]][['index','title','author_id','report_time','col','code_flag','content']]
            author_4_sorted = author_4_sorted.sort_values(by=['report_time'],ascending=False).head(number_of_results).values.tolist()
        except:
            author_4_sorted = None      
        
        #2) Authors according to  relevance 
        try:
            author_1 = authors[authors['author_id']==auths[0]][['index','title','author_id','report_time','col','code_flag','content']].head(number_of_results).values.tolist()
        except:
            author_2 = None
        try:
            author_2 = authors[authors['author_id']==auths[1]][['index','title','author_id','report_time','col','code_flag','content']].head(number_of_results).values.tolist()
        except:
            author_2 = None
        try:
            author_3 = authors[authors['author_id']==auths[2]][['index','title','author_id','report_time','col','code_flag','content']].head(number_of_results).values.tolist()
        except:
            author_3 = None
        try:
            author_4 = authors[authors['author_id']==auths[3]][['index','title','author_id','report_time','col','code_flag','content']].head(number_of_results).values.tolist()
        except:
            author_4 = None

    #stacking all authors to a list to pass as ajax response object 
    author_titles = [author_1,author_2,author_3,author_4]
    author_titles_sorted = [author_1_sorted,author_2_sorted,author_3_sorted,author_4_sorted]
    # logging.info(top_searches)
    return top_searches,author_names,author_titles,top_searches_sorted,author_titles_sorted,images,images_sorted
    # else:
    #     return None,None,None,None,None,None,None

app.debug = True



if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
