from ast import If
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
import sys
import json
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import csv
import requests as rq
from bs4 import BeautifulSoup as bs
import csv
import pandas as pd
import numpy as np
import re

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

import datetime
import pickle

import nltk

import requests
import time
# nltk.download('all')

df = pd.read_excel("label_new_1203.xlsx")
df
df['TITLE'] = df['Title']
def std_text(text):
    text = text.lower()
    text = re.sub('\s\W',' ',text)
    text = re.sub('\W\s',' ',text)
    text = re.sub('\s+',' ',text)
    return text

df['TITLE'] = [std_text(s) for s in df['TITLE']]
df['TITLE'] = [word_tokenize(s) for s in df['TITLE']]
tags = defaultdict(lambda : wn.NOUN)
tags['J'] = wn.ADJ
tags['V'] = wn.VERB
tags['R'] = wn.ADV
for index,entry in enumerate(df['TITLE']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and re.match(r'^[A-Za-z0-9_-]+$', word):
            word_Final = word_Lemmatized.lemmatize(word,tags[tag[0]])
            Final_words.append(word_Final)
    df.loc[index,'title_final'] = str(Final_words)


X_train, X_test, y_train, y_test = model_selection.train_test_split(df['title_final'],df['Category'],
                                                                    test_size=0.25, 
                                                                    random_state=2,
                                                                    stratify=df['Category'])
Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.transform(y_test)
Tfidf_vect = TfidfVectorizer(max_features=15000)
Tfidf_vect.fit(df['title_final'])
X_train = Tfidf_vect.transform(X_train)
X_test = Tfidf_vect.transform(X_test)
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb.score(X_test, y_test)
print("scpre: ",nb.score(X_test, y_test) )

# f = open('03_knews_nb.pickle', 'wb')
# pickle.dump(nb, f)
# f.close()
# f = open('03_knews_vectorizer.pickle', 'wb')
# pickle.dump(Tfidf_vect, f)
# f.close()
# f = open('03_knews_Encoder.pickle', 'wb')
# pickle.dump(Encoder, f)
# f.close()