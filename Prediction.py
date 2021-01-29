import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import pipeline
from sklearn import metrics
import pickle

df = pd.read_csv('sqi.txt', sep="\t", header=None, names=["English", "Albanian", "cite"])
#print(df)
Eng_df=df[["English"]]
Alb_df=df[["Albanian"]]
#print(Eng_df)
#print(Alb_df)
table = str.maketrans('', '', string.punctuation)
data_En=[]
lang_En=[]
data_Alb=[]
lang_Alb=[]
for i,mk in Eng_df.iterrows():
    mk=mk["English"]
    if len(mk)!=0:
        mk=mk.lower()
        mk=re.sub(r"\d+","",mk)
        mk=mk.translate(table)
        data_En.append(mk)
        lang_En.append("English")
for i,mk in Alb_df.iterrows():
    mk=mk["Albanian"]
    if len(mk)!=0:
        mk=mk.lower()
        mk=re.sub(r"\d+","",mk)
        mk=mk.translate(table)
        data_Alb.append(mk)
        lang_Alb.append("Albanian")
df=pd.DataFrame({"Text":data_En+data_Alb,
                   "Lang":lang_En+lang_Alb})
df=df.sample(frac=1)
#print(df)
x,y=df.iloc[:,0],df.iloc[:,1]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)
vector=feature_extraction.text.TfidfVectorizer(ngram_range=(1,3),analyzer="char")
pipee=pipeline.Pipeline([
    ("vectorizer",vector),
    ("clf",linear_model.LogisticRegression())
])
pipee.fit(x_train,y_train)
ypre=pipee.predict(x_test)
acc=(metrics.accuracy_score(y_test,ypre))*100
#print(acc,"%")
#matrix=metrics.confusion_matrix(y_test,ypre)
#print("confusion matrix: \n ", matrix)
LDf=open('LDmodel2.pck1','wb')
pickle.dump(pipee,LDf)
LDf.close()
