# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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




#NEWS SEARCH + NB MODEL USAGE
#This returns JSON file that contains Title, URL, Content, Writer, Date, Category, and the respective Image Link
def std_text(text):
    text = text.lower()
    text = re.sub('\s\W',' ',text)
    text = re.sub('\W\s',' ',text)
    text = re.sub('\s+',' ',text)
    return text

def preproc_df(df):
  df['TITLE'] = df['Title']
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
  return df

def preproc_str(strg):
  strg = std_text(strg)
  strg = word_tokenize(strg)
  tags = defaultdict(lambda : wn.NOUN)
  tags['J'] = wn.ADJ
  tags['V'] = wn.VERB
  tags['R'] = wn.ADV
  Final_words = []
  word_Lemmatized = WordNetLemmatizer()
  for word, tag in pos_tag(strg):
      if word not in stopwords.words('english') and re.match(r'^[A-Za-z0-9_-]+$', word):
          word_Final = word_Lemmatized.lemmatize(word,tags[tag[0]])
          Final_words.append(word_Final)
  return str(Final_words)

f = open('03_knews_nb.pickle', 'rb')
nb = pickle.load(f)
f.close()
f = open('03_knews_vectorizer.pickle', 'rb')
Tfidf_vect = pickle.load(f)
f.close()
f = open('03_knews_Encoder.pickle', 'rb')
Encoder = pickle.load(f)
f.close()




#set sec to 86400 for a day interval

sec = 86400
limit_year = 2022
retry_limit = 10
print(f"News is limited only until the year {limit_year} from now")
print(f"Interval is set to {sec} seconds")

start = time.time()
timeshow = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
print("Process of collecting ALL news begins: " + timeshow)
print("-----------------------------------------------------------------")
print("Collecting news from Soompi...")
titles = []
urls = []
contents = []
writers = []
dates = []
categories = []
imgs = []
websites= []
col = ['title','url','content','writer','date','category','img_link','website']
timenow = datetime.datetime.now().strftime("%d%m%y%H%M%S")
fname = 'all_sites_'+timenow
hasnext = "True"
pagenum = 1
retry = "False"
retry_n = 0
while hasnext=="True":
  if(retry == "True"):
    retry = "False"
  else:
    pagenum += 1
  if retry_n>retry_limit:
    print("Retry limit reached. Stopping current process... (continue to next process)")
    break
  try:
    data = rq.get('https://api-fandom.soompi.com/categories/9wpc/posts.json?perPage=&page='+str(pagenum)).json()
  except:
    retry_n +=1
    print(f"Error loading page, retrying...({retry_n})")
    retry = "True"
    continue
  retry_n = 0
  hasnext = str(data['pageInfo']['hasNextPage'])
  # print(hasnext)
  pagenum += 1
  # print(data['results'])
  for d in data['results']:
    # print(d['id'])
    # print(d['id']+'/'+d['slug'])
    url = 'https://www.soompi.com/article/'+d['id']+'/'+d['slug']
    title = d['title']['text']
    print(title)
    page = rq.get(url)
    soup = bs(page.content, 'html.parser')
    # print(soup.prettify())
    # info-emphasis right
    writer = soup.find("div",{"class":"info-emphasis right"}).find("a").decode_contents()
    date = soup.find("div",{"class":"date-time"}).decode_contents()
    datef = datetime.datetime.strptime(date, "%b %d, %Y")
    print(date)
    if(datef.year>=limit_year):
      paragraphs = soup.find("div",{"class":"article-wrapper"}).findAll("p")
      content = ''
      for p in paragraphs:
        content += bs(p.decode_contents(),'html.parser').get_text()+' '
      img_link = soup.find("span",{"class":"image-wrapper"}).find("img").get("src")
      img_link = img_link.split("?s=")[0]
      txt = [preproc_str(title)]
      X = Tfidf_vect.transform(txt)
      cat = Encoder.inverse_transform(nb.predict(X))[0]
      titles.append(title)
      urls.append(url)
      contents.append(content)
      writers.append(writer)
      dates.append(date)
      categories.append(cat)
      imgs.append(img_link)
      websites.append("soompi")
    else:
      hasnext="False"
      break
    
pagenum = 0
stop = "False"
retry = "False"
retry_n = 0
print("-----------------------------------------------------------------")
print("Collecting news from Koreaboo...")
while True:
  if(retry == "True"):
    retry = "False"
  else:
    pagenum += 1
  if retry_n>retry_limit:
    print("Retry limit reached. Stopping current process... (continue to next process)")
    break
  try:
    page = rq.get('https://www.koreaboo.com/news/page/'+str(pagenum)+'/')
  except:
    retry_n +=1
    print(f"Error loading page, retrying...({retry_n})")
    retry = "True"
    continue
  retry_n = 0
  soup = bs(page.content, 'html.parser')
  news = soup.findAll("article",{"class": "cat-news"})
  if len(news)<1:
    break
  if(stop == "True"):
    break
  for item in news:
    url = (item.find("a").get("href"))
    title = item.find("div",{"class":"ap-chron-medium-title"}).decode_contents()
    print(title)
    page = rq.get(url)
    detail = bs(page.content, 'html.parser')
    paragraphs = detail.find("div",{"class":"entry-content"}).findAll("p")
    content = ''
    for p in paragraphs:
      content += bs(p.decode_contents(),'html.parser').get_text()+' '
    if(detail.find("div",{"class":"writer-bio-name"}).find("a")):
      writer = detail.find("div",{"class":"writer-bio-name"}).find("a").decode_contents()
    else:
      writer = "Koreaboo"
    date = detail.find("div",{"class":"posted-on"}).find("time").get('datetime')[:10]
    datef = datetime.datetime.strptime(date, "%Y-%m-%d")
    print(date)
    if(datef.year>=limit_year):
      img_link = detail.find("img",{"class":"featured-image"}).get("src")
      txt = title + " " + content
      txt = [preproc_str(txt)]
      X = Tfidf_vect.transform(txt)
      cat = Encoder.inverse_transform(nb.predict(X))[0]
      titles.append(title)
      urls.append(url)
      contents.append(content)
      writers.append(writer)
      dates.append(date)
      categories.append(cat)
      imgs.append(img_link)
      websites.append("koreaboo")
    else:
      stop = "True"
      break
results_object = [dict(zip(col, row)) for row in zip(titles,urls,contents,writers,dates,categories,imgs,websites)]
for result in results_object:
  requests.post("http://localhost:3000/artist-news/", data = result)

#Collect News posted today/certain date from both site + repetition
timeshow = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
print(f"COMPLETED collecting ALL news from both sites! saved as {fname}.json")
print("Completion time: "+timeshow)
print("=================================================================")
diff = time.time()-start
sec2 = sec - diff
hour = int(int(sec2)/3600)
min = int(sec2)%3600/60
timefut = datetime.datetime.now()+datetime.timedelta(seconds=sec2)
print("Next processes will occur daily (or according to the interval set) ==> collecting news written at the corresponding day")
print(f"Next process in {sec2:.2f} seconds (~{hour} hours and {min:.1f} minutes or around "+timefut.strftime("%d-%m-%y %H:%M:%S")+")")
print("=================================================================")
time.sleep(sec2)

while(True):
  start = time.time()
  timenow = datetime.datetime.now().strftime("%d%m%y%H%M%S")
  timeshow = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
  print("Process begins " + timeshow)
  print("Collecting news from Soompi...")
  titles = []
  urls = []
  contents = []
  writers = []
  dates = []
  categories = []
  imgs = []
  websites = []
  col = ['title','url','content','writer','date','category','img_link','website']
  datenow = datetime.datetime.now().strftime("%b %-d, %Y")
  
  fname = 'soompi_today_'+timenow
  hasnext = "True"
  pagenum = 1
  stop = False
  retry = "False"
  retry_n = 0
  while hasnext=="True":
    if(retry == "True"):
      retry = "False"
    else:
      pagenum += 1
    if retry_n>retry_limit:
      print("Retry limit reached. Stopping current process... (continue to next process)")
      break
    try:
      data = rq.get('https://api-fandom.soompi.com/categories/9wpc/posts.json?perPage=&page='+str(pagenum)).json()
    except:
      retry_n +=1
      print(f"Error loading page, retrying...({retry_n})")
      retry = "True"
      continue
    retry_n = 0
    hasnext = str(data['pageInfo']['hasNextPage'])
    pagenum += 1
    # print(data['results'])
    for d in data['results']:
      # # print(d['id'])
      # print(d['id']+'/'+d['slug'])
      url = 'https://www.soompi.com/article/'+d['id']+'/'+d['slug']
      title = d['title']['text']
      page = rq.get(url)
      soup = bs(page.content, 'html.parser')
      # print(soup.prettify())
      # info-emphasis right
      writer = soup.find("div",{"class":"info-emphasis right"}).find("a").decode_contents()
      date = soup.find("div",{"class":"date-time"}).decode_contents()
      # print(date)
      datef = datetime.datetime.strptime(date, "%b %d, %Y")
      datenowf = datetime.datetime.strptime(datenow, "%b %d, %Y")
      # print(datef)
      if datef > datenowf:
        continue
      elif datef == datenowf:
        print(title)
        paragraphs = soup.find("div",{"class":"article-wrapper"}).findAll("p")
        content = ''
        for p in paragraphs:
          content += bs(p.decode_contents(),'html.parser').get_text()+' '
        img_link = soup.find("span",{"class":"image-wrapper"}).find("img").get("src")
        img_link = img_link.split("?s=")[0]
        txt = [preproc_str(title)]
        X = Tfidf_vect.transform(txt)
        cat = Encoder.inverse_transform(nb.predict(X))[0]
        titles.append(title)
        urls.append(url)
        contents.append(content)
        writers.append(writer)
        dates.append(date)
        categories.append(cat)
        imgs.append(img_link)
        websites.append("soompi")
      else:
        stop = True
        break
    if stop==True:
      break
    
  results_object = [dict(zip(col, row)) for row in zip(titles,urls,contents,writers,dates,categories,imgs,websites)]
  for result in results_object:
    requests.post("http://localhost:3000/artist-news/", data = result)
  print("-----------------------------------------------------------------")
  print("Collecting news from Koreaboo...")
  titles = []
  urls = []
  contents = []
  writers = []
  dates = []
  categories = []
  imgs = []
  websites = []
  col = ['title','url','content','writer','date','category','img_link','website']
  fname = 'koreaboo_today_'+timenow
  pagenum = 0
  datenow = datetime.datetime.now().strftime("%Y-%m-%d")
  stop = False
  retry = "False"
  retry_n = 0
  while True:
    if(retry == "True"):
      retry = "False"
    else:
      pagenum += 1
    if retry_n>retry_limit:
      print("Retry limit reached. Stopping current process... (continue to next process)")
      break
    try:
      page = rq.get('https://www.koreaboo.com/news/page/'+str(pagenum)+'/')
    except:
      retry_n +=1
      print(f"Error loading page, retrying...({retry_n})")
      retry = "True"
      continue
    retry_m = 0
    soup = bs(page.content, 'html.parser')
    news = soup.findAll("article",{"class": "cat-news"})
    if  len(news)<1:
      break
    for item in news:
      url = (item.find("a").get("href"))
      title = item.find("div",{"class":"ap-chron-medium-title"}).decode_contents()
      page = rq.get(url)
      detail = bs(page.content, 'html.parser')
      paragraphs = detail.find("div",{"class":"entry-content"}).findAll("p")
      content = ''
      for p in paragraphs:
        content += bs(p.decode_contents(),'html.parser').get_text()+' '
      if(detail.find("div",{"class":"writer-bio-name"}).find("a")):
        writer = detail.find("div",{"class":"writer-bio-name"}).find("a").decode_contents()
      else:
        writer = "Koreaboo"
      date = detail.find("div",{"class":"posted-on"}).find("time").get('datetime')[:10]
      datef = datetime.datetime.strptime(date, "%Y-%m-%d")
      datenowf = datetime.datetime.strptime(datenow, "%Y-%m-%d")
      # print(datef)
      if datef > datenowf:
        continue
      elif datef == datenowf:
        print(title)
        img_link = detail.find("img",{"class":"featured-image"}).get("src")
        txt = [preproc_str(title)]
        X = Tfidf_vect.transform(txt)
        cat = Encoder.inverse_transform(nb.predict(X))[0]
        titles.append(title)
        urls.append(url)
        contents.append(content)
        writers.append(writer)
        dates.append(date)
        categories.append(cat)
        imgs.append(img_link)
        websites.append("koreaboo")
      else:
        stop = True
        break
    if(stop==True):
      break
  results_object = [dict(zip(col, row)) for row in zip(titles,urls,contents,writers,dates,categories,imgs,websites)]
  for result in results_object:
    requests.post("http://localhost:3000/artist-news/", data = result)
  
  diff = time.time()-start
  sec2 = sec - diff
  hour = int(int(sec2)/3600)
  min = int(sec2)%3600/60
  timefut = datetime.datetime.now()+datetime.timedelta(seconds=sec2)
  print(f"Elapsed Time: {diff:.2f} seconds. Saved file as {fname}.json")
  print(f"Next process in {sec2:.2f} seconds (~{hour} hours and {min:.1f} minutes or around "+timefut.strftime("%d-%m-%y %H:%M:%S")+")")
  print("=================================================================")
  time.sleep(sec2)