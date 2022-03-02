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

#open csv koreaboo
# f = open('result_koreaboo.csv', 'w')
# w = csv.writer(f)
# w.writerow(('Title','Content','Writer','Date','URL','Category'))
#change the range parameter to collect certain page. (5,9) means it'll collect from page 5 to 8
# for i in range(21,41):
#   pagenum = i+1
#   page = rq.get('https://www.koreaboo.com/news/page/'+str(pagenum)+'/')
#   soup = bs(page.content, 'html.parser')
#   for item in soup.findAll("article",{"class": "cat-news"}):
#     url = (item.find("a").get("href"))
#     title = item.find("div",{"class":"ap-chron-medium-title"}).decode_contents()
#     print(title)
#     print(url)
#     page = rq.get(url)
#     detail = bs(page.content, 'html.parser')
#     paragraphs = detail.find("div",{"class":"entry-content"}).findAll("p")
#     content = ''
#     for p in paragraphs:
#       content += bs(p.decode_contents(),'html.parser').get_text()+' '
#     if(detail.find("div",{"class":"writer-bio-name"}).find("a")):
#       writer = detail.find("div",{"class":"writer-bio-name"}).find("a").decode_contents()
#     else:
#       writer = "Koreaboo"
#     date = detail.find("div",{"class":"posted-on"}).find("time").get('datetime')[:10]
#     print(content)
#     print(writer)
#     row = title,content,writer,date,url,''
#     w.writerow(row)
# f.close()

#open csv soompi
# f = open('result_soompi.csv', 'w')
# w = csv.writer(f)
# w.writerow(('Title','Content','Writer','Date','URL','Category'))
#change the range parameter to collect certain page. (5,9) means it'll collect from page 5 to 8
# for i in range(5,9):
#   data = rq.get('https://api-fandom.soompi.com/categories/9wpc/posts.json?perPage=50&page='+str(i+1)).json()
#   # print(data['results'])
#   for d in data['results']:
#     # print(d['id'])
#     print(d['id']+'/'+d['slug'])
#     url = 'https://www.soompi.com/article/'+d['id']+'/'+d['slug']
#     title = d['title']['text']
#     page = rq.get(url)
#     soup = bs(page.content, 'html.parser')
#     writer = soup.find("div",{"class":"info-emphasis right"}).find("a").decode_contents()
#     date = soup.find("div",{"class":"date-time"}).decode_contents()
#     paragraphs = soup.find("div",{"class":"article-wrapper"}).findAll("p")
#     content = ''
#     for p in paragraphs:
#       content += bs(p.decode_contents(),'html.parser').get_text()+' '
#     row = title,content,writer,date,url,''
#     w.writerow(row)
# f.close()

#Merging datasets + Adding image links in data + checking for duplicates
# df = pd.read_csv('1_koreaboo_label.csv',encoding="UTF-8")
# df2 = pd.read_csv('2_koreaboo_label.csv', encoding = "cp1252")
# df = pd.concat([df,df2],ignore_index=True)
# df = df.drop_duplicates(subset=['Title'])
# df['img'] = ""
# for index, row in df.iterrows():
#     print(row['URL'])
#     page = rq.get(row['URL'])
#     soup = bs(page.content, 'html.parser')
#     img_link = soup.find("img",{"class":"featured-image"}).get("src")
#     df.loc[index, 'img'] = img_link
#     print(img_link)
# df.to_csv("koreaboo_wimg_label.csv",encoding="UTF-8")

#Merging datasets + Adding image links in data + checking for duplicates
# df = pd.read_csv('1_soompi_label.csv',encoding="UTF-8")
# df2 = pd.read_csv('2_soompi_label.csv', encoding = "cp1252")
# df = pd.concat([df,df2],ignore_index=True)
# df = df.drop_duplicates(subset=['Title'])
# df['img'] = ""
# for index, row in df.iterrows():
#     page = rq.get(row['URL'])
#     soup = bs(page.content, 'html.parser')
#     img_link = soup.find("span",{"class":"image-wrapper"}).find("img").get("src")
#     df.loc[index, 'img'] = img_link
# df.to_csv("soompi_wimg_label.csv",encoding="UTF-8")

#making naive bayes model
# df['TEXT'] = df['Title']+" "+df['Content']
# def std_text(text):
#     text = text.lower()
#     text = re.sub('\s\W',' ',text)
#     text = re.sub('\W\s',' ',text)
#     text = re.sub('\s+',' ',text)
#     return text
#
# df['TEXT'] = [std_text(s) for s in df['TEXT']]
# df['TEXT'] = [word_tokenize(s) for s in df['TEXT']]
# tags = defaultdict(lambda : wn.NOUN)
# tags['J'] = wn.ADJ
# tags['V'] = wn.VERB
# tags['R'] = wn.ADV
# for index,entry in enumerate(df['TEXT']):
#     Final_words = []
#     word_Lemmatized = WordNetLemmatizer()
#     for word, tag in pos_tag(entry):
#         if word not in stopwords.words('english') and re.match(r'^[A-Za-z0-9_-]+$', word):
#             word_Final = word_Lemmatized.lemmatize(word,tags[tag[0]])
#             Final_words.append(word_Final)
#     df.loc[index,'text_final'] = str(Final_words)
# df = pd.read_csv('koreaboo_wimg_label.csv',encoding='UTF-8')
# df2 = pd.read_csv('soompi_wimg_label.csv',encoding='UTF-8')
# df = pd.concat([df,df2],ignore_index=True)
# X_train, X_test, y_train, y_test = model_selection.train_test_split(df['text_final'],df['Category'],
#                                                                     test_size=0.2,
#                                                                     random_state=0)
# Encoder = LabelEncoder()
# y_train = Encoder.fit_transform(y_train)
# y_test = Encoder.transform(y_test)
# Tfidf_vect = TfidfVectorizer(max_features=15000)
# Tfidf_vect.fit(df['text_final'])
# X_train = Tfidf_vect.transform(X_train)
# X_test = Tfidf_vect.transform(X_test)
# nb = MultinomialNB()
# nb.fit(X_train, y_train)
# nb.score(X_test, y_test)


#Save the NB model, vectorizer, and encoder
# f = open('knews_nb.pickle', 'wb')
# pickle.dump(nb, f)
# f.close()
# f = open('knews_vectorizer.pickle', 'wb')
# pickle.dump(Tfidf_vect, f)
# f.close()
# f = open('knews_Encoder.pickle', 'wb')
# pickle.dump(Encoder, f)
# f.close()


#NEWS SEARCH + NB MODEL USAGE
#This returns JSON file that contains Title, URL, Content, Writer, Date, Category, and the respective Image Link
def std_text(text):
    text = text.lower()
    text = re.sub('\s\W',' ',text)
    text = re.sub('\W\s',' ',text)
    text = re.sub('\s+',' ',text)
    return text

def preproc_df(df):
  df['TEXT'] = [std_text(s) for s in df['TEXT']]
  df['TEXT'] = [word_tokenize(s) for s in df['TEXT']]
  tags = defaultdict(lambda : wn.NOUN)
  tags['J'] = wn.ADJ
  tags['V'] = wn.VERB
  tags['R'] = wn.ADV
  for index,entry in enumerate(df['TEXT']):
      Final_words = []
      word_Lemmatized = WordNetLemmatizer()
      for word, tag in pos_tag(entry):
          if word not in stopwords.words('english') and re.match(r'^[A-Za-z0-9_-]+$', word):
              word_Final = word_Lemmatized.lemmatize(word,tags[tag[0]])
              Final_words.append(word_Final)
      df.loc[index,'text_final'] = str(Final_words)
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

f = open('knews_nb.pickle', 'rb')
nb = pickle.load(f)
f.close()
f = open('knews_vectorizer.pickle', 'rb')
Tfidf_vect = pickle.load(f)
f.close()
f = open('knews_Encoder.pickle', 'rb')
Encoder = pickle.load(f)
f.close()

#koreaboo
# #Collect ALL NEWS for a search term
# titles = []
# urls = []
# contents = []
# writers = []
# dates = []
# categories = []
# imgs = []
# website = []
# artist_name = []
# col = ['title','url','content','writer','date','category','img_link','website','artist']

# print('This cell will collect as many news as possible based on your search term.')
# search_term = input('Search Term: ')
# timenow = datetime.datetime.now().strftime("%d%m%y%H%M%S")
# fname = 'koreaboo_'+search_term+'_'+'full'+'_'+timenow
# link = 'https://search.koreaboo.com/q/'+search_term
# cursor = ''
# dataAvailable = True
# while(dataAvailable):
#   rq_link = link+cursor
#   data = rq.get(rq_link).json()
#   next_cursor = "/?cursor="+data['nextCursorMark']
#   if(next_cursor == "/?cursor=*" or cursor == next_cursor):
#     break
#   cursor = next_cursor
#   data['response']['docs']
#   for d in data['response']['docs']:
#       url = d['url']
#       title = d['title']
#       page = rq.get(url)
#       detail = bs(page.content, 'html.parser')
#       paragraphs = detail.find("div",{"class":"entry-content"}).findAll("p")
#       content = ''
#       for p in paragraphs:
#         content += (bs(p.decode_contents(),'html.parser').get_text()+' ')
#       if(detail.find("div",{"class":"writer-bio-name"}).find("a")):
#         writer = detail.find("div",{"class":"writer-bio-name"}).find("a").decode_contents()
#       else:
#         writer = "Koreaboo"
#       date = detail.find("div",{"class":"posted-on"}).find("time").get('datetime')[:10]
#       img_link = detail.find("img",{"class":"featured-image"}).get("src")
#       txt = title + " " + content
#       txt = [preproc_str(txt)]
#       X = Tfidf_vect.transform(txt)
#       cat = Encoder.inverse_transform(nb.predict(X))[0]
#       titles.append(title)
#       urls.append(url)
#       contents.append(content)
#       writers.append(writer)
#       dates.append(date)
#       categories.append(cat)
#       imgs.append(img_link)
#       website.append("koreaboo")
#       artist_name.append(search_term)
# results_object = [dict(zip(col, row)) for row in zip(titles,urls,contents,writers,dates,categories,imgs,website,artist_name)]

# for result in results_object:
#   requests.post("https://localhost:3000/artist-news/", data = result)
# json_object = json.dumps([dict(zip(col, row)) for row in zip(titles,urls,contents,writers,dates,categories,imgs,website,artist_name)], indent = 4,ensure_ascii=False)
# with open(fname+".json", "w", encoding='utf8') as outfile:
#     outfile.write(json_object)

# #soompi
# #Collect ALL NEWS for a search term
# titles = []
# urls = []
# contents = []
# writers = []
# dates = []
# categories = []
# imgs = []
# websites = []
# col = ['title','url','content','writer','date','category','img_link','website']

# print('This cell will collect as many news as possible based on your search term.')
# search_term = input('Search Term: ')
# timenow = datetime.datetime.now().strftime("%d%m%y%H%M%S")
# fname = 'soompi_'+search_term+'_'+'full'+'_'+timenow
# i=0
# while(True):
#   data = rq.get('https://api-fandom.soompi.com/search.json?query='+search_term+'&page='+str(i)+'&perPage=20&hasMore=%7B"article":false,"fanclub":false,"tag":false%7D&topicId=search&filters=type:post&lang=en').json()
#   i+=1
#   hasnext = data["pageInfo"]["hasNextPage"]
#   for d in data['results']:
#     # print(d['id']+'/'+d['slug'])
#     url = 'https://www.soompi.com/article/'+d['id']+'/'+d['slug']
#     title = d['title']
#     page = rq.get(url)
#     soup = bs(page.content, 'html.parser')
#     writer = soup.find("div",{"class":"info-emphasis right"}).find("a").decode_contents()
#     date = soup.find("div",{"class":"date-time"}).decode_contents()
#     paragraphs = soup.find("div",{"class":"article-wrapper"}).findAll("p")
#     content = ''
#     for p in paragraphs:
#       content += bs(p.decode_contents(),'html.parser').get_text()+' '
#     img_link = soup.find("span",{"class":"image-wrapper"}).find("img").get("src")
#     txt = title + " " + content
#     txt = [preproc_str(txt)]
#     X = Tfidf_vect.transform(txt)
#     cat = Encoder.inverse_transform(nb.predict(X))[0]
#     titles.append(title)
#     urls.append(url)
#     contents.append(content)
#     writers.append(writer)
#     dates.append(date)
#     categories.append(cat)
#     imgs.append(img_link)
#     websites.append("soompi")
  
#   # for d in data['results']:
#   #   print(d)
#   if(hasnext==False):
#     break

# results_object = [dict(zip(col, row)) for row in zip(titles,urls,contents,writers,dates,categories,imgs,websites)]
# for result in results_object:
#   requests.post("http://localhost:3000/artist-news/", data = result)
# json_object = json.dumps([dict(zip(col, row)) for row in zip(titles,urls,contents,writers,dates,categories,imgs,website,artist_name)], indent = 4,ensure_ascii=False)
# with open(fname+".json", "w", encoding='utf8') as outfile:
#     outfile.write(json_object)

# sec = 86400
# limit_year = 2022
# print(f"News is limited only until the year {limit_year} from now")
# print(f"Interval is set to {sec} seconds")

# start = time.time()
# timeshow = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
# print("Process of collecting ALL news begins: " + timeshow)
# print("Begin collecting news from Soompi..")
# titles = []
# urls = []
# contents = []
# writers = []
# dates = []
# categories = []
# imgs = []
# websites = []
# col = ['title','url','content','writer','date','category','img_link','website']
# timenow = datetime.datetime.now().strftime("%d%m%y%H%M%S")
# fname = 'all_sites_'+timenow
# hasnext = "True"
# pagenum = 1
# while hasnext=="True":
#   data = rq.get('https://api-fandom.soompi.com/categories/9wpc/posts.json?perPage=&page='+str(pagenum)).json()
#   hasnext = str(data['pageInfo']['hasNextPage'])
#   # print(hasnext)
#   pagenum += 1
#   # print(data['results'])
#   for d in data['results']:
#     # print(d['id'])
#     # print(d['id']+'/'+d['slug'])
#     url = 'https://www.soompi.com/article/'+d['id']+'/'+d['slug']
#     title = d['title']['text']
#     print(title)
#     page = rq.get(url)
#     soup = bs(page.content, 'html.parser')
#     # print(soup.prettify())
#     # info-emphasis right
#     writer = soup.find("div",{"class":"info-emphasis right"}).find("a").decode_contents()
#     date = soup.find("div",{"class":"date-time"}).decode_contents()
#     datef = datetime.datetime.strptime(date, "%b %d, %Y")
#     print(date)
#     if(datef.year>=limit_year):
#       paragraphs = soup.find("div",{"class":"article-wrapper"}).findAll("p")
#       content = ''
#       for p in paragraphs:
#         content += bs(p.decode_contents(),'html.parser').get_text()+' '
#       img_link = soup.find("span",{"class":"image-wrapper"}).find("img").get("src")
#       txt = title + " " + content
#       txt = [preproc_str(txt)]
#       X = Tfidf_vect.transform(txt)
#       cat = Encoder.inverse_transform(nb.predict(X))[0]
#       titles.append(title)
#       urls.append(url)
#       contents.append(content)
#       writers.append(writer)
#       dates.append(date)
#       categories.append(cat)
#       imgs.append(img_link)
#       websites.append("soompi")
#     else:
#       hasnext="False"
#       break
    
# pagenum = 0
# print("")
# print("Begin collecting news from Koreaboo..")
# while True:
#   pagenum += 1
#   page = rq.get('https://www.koreaboo.com/news/page/'+str(pagenum)+'/')
#   soup = bs(page.content, 'html.parser')
#   news = soup.findAll("article",{"class": "cat-news"})
#   stop = "False"
#   if len(news)<1:
#     break
#   if(stop == "True"):
#     break
#   for item in news:
#     url = (item.find("a").get("href"))
#     title = item.find("div",{"class":"ap-chron-medium-title"}).decode_contents()
#     print(title)
#     page = rq.get(url)
#     detail = bs(page.content, 'html.parser')
#     paragraphs = detail.find("div",{"class":"entry-content"}).findAll("p")
#     content = ''
#     for p in paragraphs:
#       content += bs(p.decode_contents(),'html.parser').get_text()+' '
#     if(detail.find("div",{"class":"writer-bio-name"}).find("a")):
#       writer = detail.find("div",{"class":"writer-bio-name"}).find("a").decode_contents()
#     else:
#       writer = "Koreaboo"
#     date = detail.find("div",{"class":"posted-on"}).find("time").get('datetime')[:10]
#     datef = datetime.datetime.strptime(date, "%Y-%m-%d")
#     print(date)
#     if(datef.year>=limit_year):
#       img_link = detail.find("img",{"class":"featured-image"}).get("src")
#       txt = title + " " + content
#       txt = [preproc_str(txt)]
#       X = Tfidf_vect.transform(txt)
#       cat = Encoder.inverse_transform(nb.predict(X))[0]
#       titles.append(title)
#       urls.append(url)
#       contents.append(content)
#       writers.append(writer)
#       dates.append(date)
#       categories.append(cat)
#       imgs.append(img_link)
#       websites.append("koreaboo")
#     else:
#       stop = "True"
#       break
# results_object = [dict(zip(col, row)) for row in zip(titles,urls,contents,writers,dates,categories,imgs,websites)]
# for result in results_object:
#   requests.post("http://localhost:3000/artist-news/", data = result)

# #Collect News posted today/certain date from both site + repetition
# timeshow = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
# print(f"COMPLETED collecting ALL news from both sites! saved as {fname}.json")
# print("Completion time: "+timeshow)
# print("=================================================================")
# diff = time.time()-start
# sec2 = sec - diff
# hour = int(int(sec2)/3600)
# min = int(sec2)%3600/60
# timefut = datetime.datetime.now()+datetime.timedelta(seconds=sec2)
# print("Next processes will occur daily (or according to the interval set) ==> collecting news written at the corresponding day")
# print(f"Next process in {sec2:.2f} seconds (~{hour} hours and {min:.1f} minutes or around "+timefut.strftime("%d-%m-%y %H:%M:%S")+")")
# print("=================================================================")
# time.sleep(sec2)

# while(True):
#   start = time.time()
#   timenow = datetime.datetime.now().strftime("%d%m%y%H%M%S")
#   timeshow = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
#   print("Process begins " + timeshow)
#   print("Collecting news from Soompi...")
#   titles = []
#   urls = []
#   contents = []
#   writers = []
#   dates = []
#   categories = []
#   imgs = []
#   websites = []
#   col = ['title','url','content','writer','date','category','img_link','website']
#   datenow = datetime.datetime.now().strftime("%b %-d, %Y")
  
#   fname = 'soompi_today_'+timenow
#   hasnext = "True"
#   pagenum = 1
#   stop = False
#   while hasnext=="True":
#     data = rq.get('https://api-fandom.soompi.com/categories/9wpc/posts.json?perPage=&page='+str(pagenum)).json()
#     hasnext = str(data['pageInfo']['hasNextPage'])
#     pagenum += 1
#     # print(data['results'])
#     for d in data['results']:
#       # # print(d['id'])
#       # print(d['id']+'/'+d['slug'])
#       url = 'https://www.soompi.com/article/'+d['id']+'/'+d['slug']
#       title = d['title']['text']
#       page = rq.get(url)
#       soup = bs(page.content, 'html.parser')
#       # print(soup.prettify())
#       # info-emphasis right
#       writer = soup.find("div",{"class":"info-emphasis right"}).find("a").decode_contents()
#       date = soup.find("div",{"class":"date-time"}).decode_contents()
#       # print(date)
#       datef = datetime.datetime.strptime(date, "%b %d, %Y")
#       datenowf = datetime.datetime.strptime(datenow, "%b %d, %Y")
#       # print(datef)
#       if datef > datenowf:
#         continue
#       elif datef == datenowf:
#         print(title)
#         paragraphs = soup.find("div",{"class":"article-wrapper"}).findAll("p")
#         content = ''
#         for p in paragraphs:
#           content += bs(p.decode_contents(),'html.parser').get_text()+' '
#         img_link = soup.find("span",{"class":"image-wrapper"}).find("img").get("src")
#         txt = title + " " + content
#         txt = [preproc_str(txt)]
#         X = Tfidf_vect.transform(txt)
#         cat = Encoder.inverse_transform(nb.predict(X))[0]
#         titles.append(title)
#         urls.append(url)
#         contents.append(content)
#         writers.append(writer)
#         dates.append(date)
#         categories.append(cat)
#         imgs.append(img_link)
#         websites.append("soompi")
#       else:
#         stop = True
#         break
#     if stop==True:
#       break
    
#   results_object = [dict(zip(col, row)) for row in zip(titles,urls,contents,writers,dates,categories,imgs,websites)]
#   for result in results_object:
#     requests.post("http://localhost:3000/artist-news/", data = result)
#   print("-----------------------------------------------------------------")
#   print("Collecting news from Koreaboo...")
#   titles = []
#   urls = []
#   contents = []
#   writers = []
#   dates = []
#   categories = []
#   imgs = []
#   websites = []
#   col = ['title','url','content','writer','date','category','img_link','website']
#   fname = 'koreaboo_today_'+timenow
#   pagenum = 0
#   datenow = datetime.datetime.now().strftime("%Y-%m-%d")
#   stop = False
#   while True:
#     pagenum += 1
#     page = rq.get('https://www.koreaboo.com/news/page/'+str(pagenum)+'/')
#     soup = bs(page.content, 'html.parser')
#     news = soup.findAll("article",{"class": "cat-news"})
#     if  len(news)<1:
#       break
#     for item in news:
#       url = (item.find("a").get("href"))
#       title = item.find("div",{"class":"ap-chron-medium-title"}).decode_contents()
#       page = rq.get(url)
#       detail = bs(page.content, 'html.parser')
#       paragraphs = detail.find("div",{"class":"entry-content"}).findAll("p")
#       content = ''
#       for p in paragraphs:
#         content += bs(p.decode_contents(),'html.parser').get_text()+' '
#       if(detail.find("div",{"class":"writer-bio-name"}).find("a")):
#         writer = detail.find("div",{"class":"writer-bio-name"}).find("a").decode_contents()
#       else:
#         writer = "Koreaboo"
#       date = detail.find("div",{"class":"posted-on"}).find("time").get('datetime')[:10]
#       datef = datetime.datetime.strptime(date, "%Y-%m-%d")
#       datenowf = datetime.datetime.strptime(datenow, "%Y-%m-%d")
#       # print(datef)
#       if datef > datenowf:
#         continue
#       elif datef == datenowf:
#         print(title)
#         img_link = detail.find("img",{"class":"featured-image"}).get("src")
#         txt = title + " " + content
#         txt = [preproc_str(txt)]
#         X = Tfidf_vect.transform(txt)
#         cat = Encoder.inverse_transform(nb.predict(X))[0]
#         titles.append(title)
#         urls.append(url)
#         contents.append(content)
#         writers.append(writer)
#         dates.append(date)
#         categories.append(cat)
#         imgs.append(img_link)
#         websites.append("koreaboo")
#       else:
#         stop = True
#         break
#     if(stop==True):
#       break
#   results_object = [dict(zip(col, row)) for row in zip(titles,urls,contents,writers,dates,categories,imgs,websites)]
#   for result in results_object:
#     requests.post("http://localhost:3000/artist-news/", data = result)
  
#   diff = time.time()-start
#   sec2 = sec - diff
#   hour = int(int(sec2)/3600)
#   min = int(sec2)%3600/60
#   timefut = datetime.datetime.now()+datetime.timedelta(seconds=sec2)
#   print(f"Elapsed Time: {diff:.2f} seconds. Saved file as {fname}.json")
#   print(f"Next process in {sec2:.2f} seconds (~{hour} hours and {min:.1f} minutes or around "+timefut.strftime("%d-%m-%y %H:%M:%S")+")")
#   print("=================================================================")
#   time.sleep(sec2)


#Collect ALL News

#set sec to 86400 for a day interval
#PAKE INI
sec = 86400
limit_year = 2022
retry_limit = 10
print(f"News is limited only until the year {limit_year} from now")
print(f"Interval is set to {sec} seconds")

start = time.time()
timeshow = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
print("Process of collecting ALL news begins: " + timeshow)
titles = []
urls = []
contents = []
writers = []
dates = []
categories = []
imgs = []
websites = []
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
      websites.append("soompi")
    else:
      hasnext="False"
      break
    
pagenum = 0
stop = "False"
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