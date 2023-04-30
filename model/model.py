import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns # for graphing 

import nltk
from nltk.corpus import stopwords
'''
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk import ngrams
'''
'''
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
'''
#import keras 
'''
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing import sequence
'''
from keras.models import load_model
from keras.preprocessing.text import Tokenizer

#import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from wordcloud import WordCloud, STOPWORDS
'''
from bs4 import BeautifulSoup
from string import punctuation
from collections import Counter
'''
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

nltk.download("stopwords")

# removes all the STOPWORDS that the gensim library and nltk library have
def preprocessing(text):
    print("running preprocessing")

    stop_words = stopwords.words("english") # gets the english words
    result = []
    # simple_proprocess convert the text into a list of tokens
    #print(gensim.utils.simple_preprocess(text))
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)


    global word_count 
    word_count = len(result)
    result = ' '.join(result)
    #print(result)

    res_list = []
    res_list.append(result)

    return res_list

def tokenize(cleaned_text):
    print("starting tokenization")
    #total_words = len(cleaned_text)
    total_words = 108705 # from the notebook
    #print(cleaned_text)

    #creates the tokenizer 
    tokenizer = Tokenizer(num_words = 400)
    tokenizer.fit_on_texts(cleaned_text)
    tokenized_res = tokenizer.texts_to_sequences(cleaned_text)
    #print(tokenized_res)
    return tokenized_res

def add_padding(tokenized_text):
    print("adding padding")
    padded_text = pad_sequences(tokenized_text, maxlen = word_count, padding="post")
    #print(padded_text)
    return padded_text

def call_model(text):
    proc_text = add_padding(tokenize(preprocessing(text)))
    model = load_model("model/misinformation_model_v2.h5")
    result = model.predict(proc_text)
    return result

#test = input()
#print(len(preprocessing(test)))
#print(tokenize(preprocessing(test)))
#add_padding(tokenize(preprocessing(test)))

#call_model(test)
#print(call_model(test))