

import numpy as np
import pandas as pd
!pip install pyLDAvis==2.1.2
!pip install nltk
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import pyLDAvis
import pyLDAvis.sklearn
from nltk.corpus import stopwords
!pip install textdistance
import textdistance

def chan(a):
  a1=".1234567890+-%=(){}_'""/,|@#$^&*`~"
  # a = sd[0]
  if isinstance(a,str):
    k = a.split()
    t=[]
    for i in k:
      if isinstance(i,str):
        t.append(i)
    for i in t:
      sum1=0
      for a in range(0,len(i)):
        if i[a] in a1:
          sum1+=1
      if sum1>0:
        t.remove(i)
    y=[]
    for i in t:
      if i not in y:
        y.append(i)
    z=[]
    for i in y:
      if len(i)>4:
        z.append(i)
    st=""
    for i in z:
      st+=i.lower()+" "
    return(st)
  else:
    return ""

def get_results(a,b,n_topics):
  df = pd.read_csv(a)
  df2 = pd.read_csv(b)
  p=[]
  for j in range(0,len(df2["Technical Abstract"])):
    p.append( df2["Proposal Number"][j]) 
  u=[]
  a1=[]
  a2=[]
  a3=[]
  for j in range(0,len(df2["Technical Abstract"])):
    l=[]
    for i in range(0,len(df["Subtopic"])):
        if df["Subtopic"][i][:6] in df2["Proposal Number"][j]:
          l.append(df["Scope Description"][i])
          a1.append(df["Scope Description"][i])
          l.append(df2["Technical Abstract"][j])
          a2.append(df2["Technical Abstract"][j])
          a3.append(df2["Proposal Number"][j])
          u.append(l)
        elif df["Subtopic"][i][:5] in df2["Proposal Number"][j]:
          l.append(df["Scope Description"][i])
          a1.append(df["Scope Description"][i])
          l.append(df2["Technical Abstract"][j])
          a2.append(df2["Technical Abstract"][j])
          a3.append(df2["Proposal Number"][j])
          u.append(l)
        else:
          pass
  df4 = pd.DataFrame({"Problem Statements":a1,"Technical Abstract":a2})
  # Reading the dataset
  data = df4

  # Preprocessing the text data
  stop_words = stopwords.words('english')
  data['Problem Statements'] = data['Problem Statements'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
  data['Problem Statements'] = data['Problem Statements'].apply(lambda x: x.lower())  # Convert to lowercase
  data['Problem Statements'] = data['Problem Statements'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))  # Remove special characters
  data['Problem Statements'] = data['Problem Statements'].apply(lambda x: x.split())  # Tokenize text
  data['Problem Statements'] = data['Problem Statements'].apply(lambda x: [i for i in x if i not in stop_words])  # Remove stop words
  data['Problem Statements'] = data['Problem Statements'].apply(lambda x: " ".join(x))  # Join the tokens back into a string

  # Creating the document-term matrix
  cv = CountVectorizer(stop_words='english')
  data_cv = cv.fit_transform(data['Problem Statements'])

  # Fitting LDA model
  lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
  gamma = lda.fit_transform(data_cv)
  beta = lda.components_
  beta1=[]
  p="a"+str(1)
  for i in beta:
    a=[]
    a.append(str(p))
    a.append(i)
    beta1.append(a)
    p="a"+str(int(p[1])+1)
  final=[]
  for i in range(0,len(gamma)):
    p=[]
    for k in range(0,len(gamma[i])):
      st=""
      for y in range(0,len(beta1[1][0])):
        for h in range(0,len(beta1)):
          for o in range(0,int((beta1[h][1][y])*gamma[i][k]*100)):
            st+=beta1[h][0]
      p.append(st)
    final.append(p)
  t1=[]
  for i in range(len(final)):
    st=""
    for j in range(len(final[i])):
      st+=final[i][j]
    t1.append(st)
  df5 = pd.DataFrame({"Technical Abstract":a2})
  # Reading the dataset
  a = df5.dropna()
  data = a

  # Preprocessing the text data
  stop_words = stopwords.words('english')
  cv = CountVectorizer(stop_words='english')
  data_cv1 = cv.fit_transform(data['Technical Abstract'])
  gamma1 = lda.fit_transform(data_cv1)
  final1=[]
  for i in range(0,len(gamma1)):
    p=[]
    for k in range(0,len(gamma1[i])):
      st=""
      for y in range(0,len(beta1[1][0])):
        for h in range(0,len(beta1)):
          for o in range(0,int((beta1[h][1][y])*gamma1[i][k]*100)):
            st+=beta1[h][0]
      p.append(st)
    final1.append(p)
  t2=[]
  for i in range(len(final1)):
    st=""
    for j in range(len(final1[i])):
      st+=final1[i][j]
    t2.append(st)
  t4=[]
  for i in range(0,len(t2)):
    t4.append(textdistance.levenshtein(t2[i],t1[i]))
  df6=pd.DataFrame({"Proposal Number":a3[:len(t4)],"Similarity":t4})
  df6.to_csv("result.csv", index=False)
  return df6.head()

get_results("2021_problemstatements.csv","2021_proposals.csv",2)

