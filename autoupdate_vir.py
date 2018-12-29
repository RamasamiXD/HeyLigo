from nltk.stem.snowball import SnowballStemmer
from sklearn.externals import joblib
from bs4 import BeautifulSoup as bs
from gensim.models import word2vec
from nltk.corpus import stopwords
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
import pickle
from math import log
import datetime as dt
import pandas as pd 
import numpy as np
import requests
import logging
import math
import csv
import re
import os

#%%

url = "https://logbook.virgo-gw.eu/virgo/index.php?sp=%s"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
stemmer = SnowballStemmer("english")
reports = []
comments = []

#%%

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

#%%

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

#%%

def extract_info(report):
    """ Helper Funtion to extract info from the page and add it to a dictionary"""
    global reports
    
    def extract_comments(coms,pid):
        global comments 
        """ Helper funtion for extracting info from commets and adding them to a list"""
        if coms:
            ids = [i.get('id')[8:] for i in coms.findAll('div',{'class':'comment'})]
            for i in ids:
                #print("-------------Comment_id: ",i)
                comment = {}
                comment['parent_id'] = pid
                comment['id'] = i
                content =  coms.findAll('div',id=re.compile(i+'$'))
                temp = content[0].text.split()
                comment['author_id']= temp[-8]
                comment['comment_time'] = dt.datetime.strptime(" ".join(temp[-6:-1]),'%H:%M, %A %d %B %Y')
                comment['content'] = " ".join(content[1].text.split())
                if len(content)>2:
                    comment['file_urls'] = [i.get('href') for i in content[3].findAll('a')]
                    comment['files'] = len(comment['file_urls'])
                else:
                    comment['file_urls'] = None 
                    comment['files'] = None 
                comments.append(comment)
            return len(ids)
        else:
            return 0
        
    data = {}
    data['id'] = report.get('id')[7:]
    #print(data['id'])
    try:
        data['section'] = " ".join(report.find(id='sectTask_'+data['id']).text.split())            
        temp = report.find(id='authHdr_'+data['id']).text.split()
        data['author_id'] = temp[-8]
        data['report_time'] = dt.datetime.strptime(" ".join(temp[-6:-1]),'%H:%M, %A %d %B %Y')
        data['title'] = " ".join(report.find(id='titleHdr_'+data['id']).text.split())
        data['content'] = " ".join(report.find(id='repHdr_'+data['id']).text.split())
        data['comments'] = extract_comments(report.find(id='comment_block_'+data['id']),data['id'])
        files = report.find(id='files_1_' + data['id'])
        if files:
            data['file_urls'] = [i.get('href') for i in files.findAll('a')]
            data['files'] = len(data['file_urls'])
        else:
            data['file_urls'] = None 
            data['files'] = None 
        reports.append(data)
    except Exception as e:
        print(e,":Error occured at",data['id'])
        pass    

#%%

def updateCSV():
    comm_df = pd.read_csv('Virgo/Original_FComments.csv',index_col=0,encoding='utf-8')
    pro_df = pd.read_csv('Virgo/Original_ProFreports.csv',index_col=0,encoding='utf-8')
    pro_df = pro_df.replace(np.nan, '', regex=True)
    comm_df = comm_df.replace(np.nan, '', regex=True)
    logging.getLogger().setLevel(logging.INFO)
    count = 1
    r = requests.get(url%count)
    soup = bs(r.text,"html.parser")
    l = soup.findAll("div",id=re.compile('^report_'))
    extract_info(l[0])
    new_posts = int(reports[0]['id']) - pro_df.sort_index(ascending=False).index.values[0]
    print("Number of new posts",new_posts)

    for page in range(1,10000):
        if len(reports)>50+new_posts:
            break
        try:
            r = requests.get(url%page)
        except:
            break
        soup = bs(r.text,"html.parser")
        l = soup.findAll("div",id=re.compile('^report_'))
        for i in l:
            extract_info(i)
        print(page)
        logging.info("Number of reports : %s"%len(reports))
        logging.info("Number of comments : %s"%len(comments))
        logging.info("Page %s parsing complete"%page)
        reps = pd.DataFrame(data=reports)
        reps.drop_duplicates('id',inplace=True)
        reps = reps.set_index('id',verify_integrity=True)
        reps.index.names = [None]
           #reps.sort_values(by='report_time',inplace=True,ascending=False)
        coms = pd.DataFrame(data=comments)
        coms.drop_duplicates('id',inplace=True)
        coms = coms.set_index('id',verify_integrity=True)
        coms.index.names = [None]
            #reps.to_csv('auto_update_test/reports.csv')
            #coms.to_csv('auto_update_test/comments.csv')
           #coms.sort_values(by='comment_time',inplace=True,ascending=False)
        logging.info("Succesfully saved till page %s"%page)  

    print("Shape of Reps",reps.shape)
    print("Shape of comments",coms.shape)
    reps.index = reps.index.astype('int64')

    reps['file_urls'].fillna(value = "",inplace=True)
    reps['files'] = reps['files'].replace(np.nan,0,regex=True)
    reps['file_urls'] = reps['file_urls'].apply(lambda x:",".join(x))
    reps['content_title'] = reps['title']+" "+reps['content']
    reps['content_title_stems'] = reps['content_title'].apply(lambda x:tokstopandstem(x))
    reps['title_stems'] = reps['title'].apply(lambda x:tokstopandstem(x))
    reps['title_authors_stems'] = (reps['author_id'].apply(lambda x:" ".join(x.split("."))) + " "+ reps['title']).apply(lambda x:tokstopandstem(x))
    reps['author_id_content_title_stems'] = (reps['author_id'].apply(lambda x:" ".join(x.split("."))) + " "+ reps['content_title']).apply(lambda x:tokstopandstem(x))


    pro_df = reps.append(pro_df)
    pro_df = pro_df.reset_index().drop_duplicates(subset='index').set_index('index')
    del pro_df.index.name
    print("Number of new posts added: ",pro_df.shape[0])
    pro_df.to_csv('Virgo/Original_ProFreports.csv',encoding='utf-8')

    comm_df =  coms.append(comm_df)
    comm_df = comm_df.reset_index().drop_duplicates(subset='index').set_index('index')
    del comm_df.index.name
    comm_df.to_csv('Virgo/Original_FComments.csv',encoding='utf-8')

    print("Final Shape of Reports Frame ",pro_df.shape)
    print("FInal shape of Comments Frame ",comm_df.shape)

#%%

def createGensim():
    pro_df = pd.read_csv('Virgo/Original_ProFreports.csv',index_col=0,encoding='utf-8')
    pro_df = pro_df.replace(np.nan, '', regex=True)
    print("Shape of data frame:",pro_df.shape)
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
    model_name = "Virgo/all_titles_authors_model"
    model.save(model_name)
    model_name = "Virgo/all_titles_authors_model"
    model = word2vec.Word2Vec.load(model_name)
    vocab = set(model.wv.index2word)
    title_author_mat = np.array(list(pro_df['title_authors_stems'].apply(lambda x:get_feature_vecs(x,model,vocab,500))))
    normalised = np.sqrt((title_author_mat*title_author_mat).sum(axis = 1))
    joblib.dump(normalised,'Virgo/normalised.pkl')
    joblib.dump(title_author_mat,'Virgo/title_author_mat.pkl')

#%%

def createTFIDF():
    pro_df = pd.read_csv('Virgo/Original_ProFreports.csv',index_col=0,encoding='utf-8')
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
    
    
    joblib.dump(words,'Virgo/words.pkl')
    joblib.dump(idf,'Virgo/idf.pkl')
    joblib.dump(norm,'Virgo/norm.pkl')
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

    # print(document_frequency)
    # print(documents_term_freq)

    count = 0
    for doc_id in documents_term_freq:
        count += 1
        if count % 5000 == 0:
            print(count)
        for word in documents_term_freq[doc_id]:
            documents_term_freq[doc_id][word] = log(1 + documents_term_freq[doc_id][word]) * log(len(sentences) / document_frequency[word])

    # print("hello1")

    rows = []
    cols = []
    data = []
    # document_vectors = np.zeros(shape = (len(documents_term_freq), word_id))
    for doc_id in range(0, len(documents_term_freq)):
        for word in documents_term_freq[doc_id]:
            rows.append(doc_id)
            cols.append(word_to_wordID[word])
            data.append(documents_term_freq[doc_id][word])
            # document_vectors[doc_id][word_to_wordID[word]] = documents_term_freq[doc_id][word]

    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)
    document_vectors = csc_matrix((data, (rows, cols)), shape = (number_of_documents, len(word_to_wordID)))

    # print("hello2")
    documents_to_concepts, concepts, concepts_to_words = svds(document_vectors, k = 500)
    i = np.eye(concepts.shape[0])
    for j in range(0,concepts.shape[0]):
        i[j][j] = concepts[j]
    documents_to_concepts = np.matmul(documents_to_concepts,i)

    # print("hello3")

    word_to_wordID_file = open("Virgo/Virgo_word_to_wordID_file", "wb")
    pickle.dump(word_to_wordID, word_to_wordID_file)

    documents_to_concepts_file = open("Virgo/Virgo_documents_to_concepts_file", "wb")
    pickle.dump(documents_to_concepts, documents_to_concepts_file)

    concepts_file = open("Virgo/Virgo_concepts_file", "wb")
    pickle.dump(concepts, concepts_file)

    concepts_to_words_file = open("Virgo/Virgo_concepts_to_words_file", "wb")
    pickle.dump(concepts_to_words, concepts_to_words_file)

#%%

def check_comments():
    fdata = pd.read_csv('Virgo/Original_FComments.csv',encoding='utf-8')
    clist = []
    for id in range(len(fdata)):
        file_urls = fdata['file_urls'].fillna('')[id].split()
        for i in range(len(file_urls)):
            if ".py" in file_urls[i]:
                clist.append(id)
                #print("found .py in {}".format(id))   
            if ".m" in file_urls[i]:
                if not (".mov" in file_urls[i] or  ".mp4" in file_urls[i] or ".mat" in file_urls[i] ) :  
                    clist.append(id)
                    #print("found .m in {} : {}".format(id,file_urls[i]))  
    alogIDs = fdata['parent_id'][clist]                
    alogIDs = np.unique(alogIDs)
    return (alogIDs)

#%%

def check_code_link():
    fname = 'Virgo/ProFreports.csv'
    os.remove(fname) if os.path.exists(fname) else None
    data = pd.read_csv('Virgo/Original_ProFreports.csv',encoding='utf-8')
    data.replace(np.nan, '', regex=True)
    out = csv.writer(open(fname,'w',encoding='utf-8'),delimiter=',',quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
    out.writerow(['','author_id','author_id_content_title_stems','comments','content','content_title','content_title_stems','file_urls','files','report_time','section','title','title_authors_stems','title_stems','code_flag'])
    pdata = data.copy(deep=True)
    codes_list = []
    for id in range(len(data)):
        #file_urls = data['file_urls'].fillna('')[id].split()
        content = data['content'].fillna('')[id].split()
        for i in range(len(content)):
            if ".py" in content[i]:
                codes_list.append(id)
                #print("found .py in {}".format(id))   
            if ".m" in content[i]:
                if not (".mov" in content[i] or  ".mp4" in content[i] or ".mat" in content[i] ) :  
                    codes_list.append(id)
                    #print("found .m in {} : {}".format(id,file_urls[i]))  
    alog_IDs = data['Unnamed: 0'][codes_list]                
    alog_IDs = np.unique(alog_IDs)
    if os.path.exists('Virgo/Original_FComments.csv'):
        comm_ID=check_comments()
        x = np.asarray([item for item in comm_ID if item not in alog_IDs])
        alog_list = np.concatenate((alog_IDs,x), axis=0)
    else:
        alog_list = alog_IDs   
    for i, row in enumerate(pdata.values):
        ids, authorid, authorstems, comments, content, contenttitle, contenttitlestems, fileurls, files, eporttime, section, title, titleauthorsstems, titlestems = row
        try:
            for item in alog_list:
                code_flag = ' '
                if ids == item:
                    code_flag = 'code mentioned'
                    break;
        except:
            code_flag = ' '
            pass
        out_data = list(row)
        out_data.append(code_flag)
        out.writerow(out_data)

#%%
     
if __name__ == "__main__":
    updateCSV()
    createGensim()
    createTFIDF()
    check_code_link()
