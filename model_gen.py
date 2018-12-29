import pandas as pd 
pd.set_option('display.max_colwidth', -1)
import numpy as np
import re, csv
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import word2vec
from nltk.stem.snowball import SnowballStemmer
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

pro_df = pd.read_csv('Livingston/ProFreports.csv',index_col=0)
##Making gensim model setting its parameters 
sentences = list(pro_df['title_authors_stems'].apply(lambda x:x.split()))
num_features_per_vec = 500
context_words = 20
num_workers = 6
downsampling = 0
min_word_count = 4
model = word2vec.Word2Vec(sentences,workers = num_workers,window=context_words,sample = downsampling, \
                 size=num_features_per_vec,min_count = min_word_count,negative=0)

model.init_sims(replace=True)
#It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "all_titles_authors_model"
model.save('Livingston/{}'.format(model_name))
model = word2vec.Word2Vec.load('Livingston/{}'.format(model_name))
vocab = set(model.wv.index2word)
print(len(vocab))

title_author_mat = np.array(list(pro_df['title_authors_stems'].apply(lambda x:get_feature_vecs(x,model,vocab,500))))
from sklearn.neighbors import NearestNeighbors
num_neighbors = int(len(vocab)/5)
nbrs = NearestNeighbors(n_neighbors=num_neighbors,algorithm='auto').fit(title_author_mat)
joblib.dump(nbrs,'Livingston/all_titles_authors_nbrs.pkl')
