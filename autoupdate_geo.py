from nltk.stem.snowball import SnowballStemmer
from selenium.webdriver.common.keys import Keys
from time import sleep
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.common.exceptions import NoSuchElementException
from sklearn.externals import joblib
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from bs4 import BeautifulSoup as bs
from gensim.models import word2vec
from nltk.corpus import stopwords
from selenium import webdriver
import datetime as dt
from math import log
import pandas as pd
import numpy as np
import requests
import pickle
import math
import csv
import re
stemmer = SnowballStemmer("english")
def get_feature_vecs(text,model,vocab,num_features):
    text = text.split()
    feature_vec = np.zeros((num_features,),dtype='float32')
    nwords = 0
    for word in text:
        if word in vocab:
            nwords=nwords+1
            feature_vec = np.add(feature_vec,model[word])
    feature_vec = np.divide(feature_vec,nwords)
    return feature_vec 
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
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  raw_html = re.sub(cleanr, '', raw_html)
  cleanr = re.compile('\n+')
  cleantext = re.sub(cleanr, '\n', raw_html)
  return cleantext
def updateCSV():
    _browser_profile = webdriver.FirefoxProfile()
    _browser_profile.set_preference("dom.webnotifications.enabled", False)
    browser = webdriver.Firefox(firefox_profile=_browser_profile)
    pro_df = pd.read_csv('Geo/ProFreports.csv',encoding='utf-8')
    pro_df = pro_df.replace(np.nan, '', regex=True)
    pages = set(pro_df['content'])
    url = "https://intranet.aei.uni-hannover.de/geo600/geohflogbook.nsf/($All)?OpenView"
    browser.get(url)
    username = browser.find_element_by_id("user-id") #username form field
    password = browser.find_element_by_id("pw-id") #password form field
    username.send_keys("reader")
    password.send_keys("readonly")
    button = browser.find_element_by_xpath("//input[@type='submit']")
    button.click()  
    r = browser.page_source
    cookiest = browser.get_cookies()
    cookies = {cookiest[0]['name']:cookiest[0]['value']}
    soup = bs(r,"html.parser")
    soup = soup.findChildren('table')[1]
    rows = soup.findChildren(['tr'])[1:]
    df = []
    i = 0
    browser.close()
    data = []
    for row in rows:
        i += 1
        col = row.findChildren(['td'])
        try:
            url = "https://intranet.aei.uni-hannover.de/"+col[2].findChildren('a')[0]['href']
        except:
            continue
        if url in pages:
                break
        print(str(i)+"/"+str(len(rows)),end="\r")
        r = requests.get(url,cookies = cookies)
        soup = bs(r.content,"html.parser")
        try:
            Content = cleanhtml(str(soup.find('form')).split("</table>")[3])
        except:  
            Content = ""
        Author = col[3].string
        Title = col[2].string
        Page = cleanhtml(str(soup.findAll('table')[1].findAll('font')[2]))
        pages.add(url)
        Date = cleanhtml(str(soup.findAll('table')[1].findAll('font')[6]))
        Time = Date.split(' ')
        Date = Time[0].split('/')
        try:
            Time = Time[1]+":00 "+Time[2]
            ContentTitle = Title + " " + Content
        except:
            continue
        Date = Date[1]+"-"+Date[0]+"-"+Date[2]+" "+Time
        Date = dt.datetime.strptime(Date,'%d-%m-%Y %H:%M:%S %p')
        ContentTitleStems = tokstopandstem(ContentTitle)
        TitleStems = tokstopandstem(Title)
        Title_Authors_Stems = TitleStems +" "+ ContentTitleStems
        try:
            AuthorContentTitleStems = tokstopandstem(Author) +" "+ tokstopandstem(ContentTitle)
        except:
            print(Author)
            AuthorContentTitleStems = ""
        data.append([Page,Author,0,url,'',0,Date,Title,ContentTitle,ContentTitleStems,TitleStems,Title_Authors_Stems,Title_Authors_Stems,'',Title])
        print("Added: ",Page)
    if len(data) > 0:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            new_df = pd.DataFrame()
            new_df = new_df.append(data)
            col = {}
            i = 0
            for val in ['','author_id','comments','content','file_urls','files','report_time','title','content_title','content_title_stems','title_stems','title_authors_stems','author_id_content_title_stems','code_flag','title_enhanced']:
                col[i] = val
                i += 1
            pro_df.rename(columns = {'Unnamed: 0':''},inplace = True)
            new_df.rename(columns = col,inplace = True)
            print(new_df.columns,pro_df.columns)
            new_df = new_df.append(pro_df)
            new_df.to_csv('Geo/ProFreports.csv',encoding='utf-8',index=False)
def createGensim():
    #pro_df = pd.read_csv('Geo/ProFreports.csv',index_col=0,encoding='utf-8', keep_default_na=False)  
    pro_df = pd.read_csv('Geo/ProFreports.csv',index_col=0,encoding='utf-8')  
    pro_df = pro_df.replace(np.nan, '', regex=True)
    sentences = list(pro_df['title_authors_stems'].apply(lambda x:x.split()))
    num_features_per_vec = 500
    context_words = 20
    num_workers = 6
    downsampling = 0
    min_word_count = 0
    model = word2vec.Word2Vec(sentences,workers = num_workers,window=context_words,sample = downsampling, \
                     size=num_features_per_vec,min_count = min_word_count,negative=0)
    model.init_sims(replace=True)
    #It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "Geo/all_titles_authors_model"
    model.save(model_name)
    model_name = "Geo/all_titles_authors_model"
    model = word2vec.Word2Vec.load(model_name)
    vocab = set(model.wv.index2word)
    title_author_mat = np.array(list(pro_df['title_authors_stems'].apply(lambda x:get_feature_vecs(x,model,vocab,500))))
    normalised = np.sqrt((title_author_mat*title_author_mat).sum(axis = 1))
    joblib.dump(normalised,'Geo/normalised.pkl')
    joblib.dump(title_author_mat,'Geo/title_author_mat.pkl')
def createTFIDF():
    pro_df = pd.read_csv('Geo/ProFreports.csv',index_col=0,encoding='utf-8')
    pro_df = pro_df.replace(np.nan, '', regex=True)
    sentences = []
    i = 0
    for line in pro_df['author_id_content_title_stems']:
        toadd = str(line).split(" ")+str(pro_df.iloc[i]['title_authors_stems']).split(" ")+str(pro_df.iloc[i]['title_authors_stems']).split(" ")+str(pro_df.iloc[i]['title_authors_stems']).split(" ")
        sentences.append(toadd)
        i+=1
    words = {}
    idf ={}
    norm = {}
    i = 0
    for sentence in sentences:
        print("\r" + str(i+1) + " / " + str(len(sentences)),end="\r")
        wordlist = set()
        for word in sentence:
            if word not in stopwords.words('english'):
                if word not in words:
                    words[word] = {}
                    idf[word] = 0
                if i not in words[word]:
                    words[word][i] = 0
                words[word][i] += 1
                wordlist.add(word)
        for word in wordlist:
            idf[word] += 1
        i += 1
    for word in idf:
        idf[word] = math.log10(len(sentences)/idf[word])
    i = 0
    for sentence in sentences:
        print("\r" + str(i+1) + " / " + str(len(sentences)),end="\r")
        norm[i] = 0
        wordlist = set()
        for word in sentence:
            if word not in stopwords.words('english'):
               wordlist.add(word)
        for word in wordlist:
            words[word][i] = (1+math.log10(words[word][i]))*idf[word]
            norm[i] += words[word][i]*words[word][i]
        norm[i] = math.sqrt(norm[i])
        i += 1
    
    
    joblib.dump(words,'Geo/words.pkl')
    joblib.dump(idf,'Geo/idf.pkl')
    joblib.dump(norm,'Geo/norm.pkl')
    number_of_documents = len(sentences)
    documents_term_freq = {}
    document_frequency = {}
    document_flag = {}
    count = 0
    word_to_wordID = {}
    word_id = 0
    for doc, doc_id in zip(sentences, range(0, len(sentences))):
        count += 1
        if count % 5000 == 0:
            print(count)
        documents_term_freq[doc_id] = {}
        for word in document_flag:
            document_flag[word] = 0
        for word in doc:
            if word not in word_to_wordID:
                word_to_wordID[word] = word_id
                word_id += 1
            if word not in document_frequency:
                document_frequency[word] = 1
            elif document_flag[word] == 0:
                document_frequency[word] += 1
            document_flag[word] = 1
            if word not in documents_term_freq[doc_id]:
                documents_term_freq[doc_id][word] = 1
            else:
                documents_term_freq[doc_id][word] += 1
    count = 0
    for doc_id in documents_term_freq:
        count += 1
        if count % 5000 == 0:
            print(count)
        for word in documents_term_freq[doc_id]:
            documents_term_freq[doc_id][word] = log(1 + documents_term_freq[doc_id][word]) * log(len(sentences) / document_frequency[word])
    rows = []
    cols = []
    data = []
    for doc_id in range(0, len(documents_term_freq)):
        for word in documents_term_freq[doc_id]:
            rows.append(doc_id)
            cols.append(word_to_wordID[word])
            data.append(documents_term_freq[doc_id][word])
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)
    document_vectors = csc_matrix((data, (rows, cols)), shape = (number_of_documents, len(word_to_wordID)))
    documents_to_concepts, concepts, concepts_to_words = svds(document_vectors, k = 500)
    i = np.eye(concepts.shape[0])
    for j in range(0,concepts.shape[0]):
        i[j][j] = concepts[j]
    documents_to_concepts1 = np.matmul(documents_to_concepts,i)
    word_to_wordID_file = open("Geo/Geo_word_to_wordID_file", "wb")
    pickle.dump(word_to_wordID, word_to_wordID_file)
    documents_to_concepts_file = open("Geo/Geo_documents_to_concepts_file", "wb")
    pickle.dump(documents_to_concepts, documents_to_concepts_file)
    concepts_file = open("Geo/Geo_concepts_file", "wb")
    pickle.dump(concepts, concepts_file)
    concepts_to_words_file = open("Geo/Geo_concepts_to_words_file", "wb")
    pickle.dump(concepts_to_words, concepts_to_words_file)
if __name__ == "__main__":
    updateCSV()
    createGensim()
    createTFIDF()
