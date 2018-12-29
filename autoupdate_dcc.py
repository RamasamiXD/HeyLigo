from nltk.stem.snowball import SnowballStemmer
from sklearn.externals import joblib
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import pandas as pd 
import requests
import math
import os
import re
import csv

#%%

stemmer = SnowballStemmer("english")
currentpath = os.getcwd()

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
    return stems

#%%

def downloadDCC():
    try:
        pro_df = pd.read_csv(currentpath+'/DCC/dcc.csv')
        pro_df = pro_df.replace(np.nan, '', regex=True)

        links = set(pro_df['link'])    
    except:
        links = set()     
    documents = []
    cleanr = re.compile('[^a-zA-Z0-9]')
    dcclink = "https://dcc.ligo.org/cgi-bin/DocDB/ListBy?topicid="
    for k in range(1,5):
        print("Downloading topic ",k)
        pageResponse = requests.get(dcclink+str(k),timeout=1000)
        pageContent = BeautifulSoup(pageResponse.content, "html.parser")
        docid = pageContent.find_all("td",{"class":"Docid"})
        title = pageContent.find_all("td",{"class":"Title"})
        author = pageContent.find_all("td",{"class":"Author"})
        i = 0
        for doc in docid:
            Link = doc.find("a")["href"]
            if Link in links:
                i += 1
                continue
            docID = re.sub(cleanr, ' ', doc.text)
            Title = re.sub(cleanr, ' ', title[i].text)
            Author = re.sub(cleanr, ' ', author[i].text)
            pageResponse2 = requests.get("https://dcc.ligo.org"+Link,timeout=1000)
            pageContent2 = BeautifulSoup(pageResponse2.content, "html.parser")
            Abstract = pageContent2.find("div",{"id":"Abstract"}).find("dd").text
            links.add(Link)
            documents.append([Link,docID,Title,Author,Abstract])
            i += 1
        try:
            with open(currentpath+"/DCC/DCC.csv", 'a') as DCCFile:
                print("Starting to write into CSV topic", k)
        except:
            with open(currentpath+"/DCC/DCC.csv", 'w') as DCCFile:
                print("File doesn't Exist! Creating CSV")
                writer = csv.DictWriter(DCCFile, fieldnames = ["link", "Ligo-Number", "Title", "Author","Abstract"],lineterminator = '\n')
                writer.writeheader()
                print("Starting tp write into CSV topic ", k)
        with open(currentpath+"/DCC/DCC.csv", 'a') as DCCFile:
            writer = csv.writer(DCCFile,lineterminator = '\n')
            writer.writerows(documents)
            print("Finished writing into CSV topic ", k)
        documents = []
    print("Updation Completed!")
#%%    
        
def createTFIDF():
    pro_df = pd.read_csv(currentpath+'/DCC/dcc.csv', keep_default_na=False)
    sentences = []
    i = 0
    print("Indexing CSV File")
    for line in pro_df['Abstract']:
        print("\r" + str(i+1),'/',len(pro_df),end="\r")
        toadd = tokstopandstem(str(line))
        sentences.append(toadd)
        i+=1
    words = {}
    idf ={}
    norm = {}
    i = 0
    print("Creating TFIDF table")
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
    print("Normalising Vectors")
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
    print("Writing Data")
    joblib.dump(words,currentpath+'/DCC/dccwords.pkl')
    joblib.dump(idf,currentpath+'/DCC/dccidf.pkl')
    joblib.dump(norm,currentpath+'/DCC/dccnorm.pkl')
    print("TFIDF table Completed")
    
#%%    MAIN

if __name__ == '__main__':
    downloadDCC()
    createTFIDF() 