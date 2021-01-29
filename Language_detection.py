import pickle
import re
import string
table = str.maketrans('', '', string.punctuation)

def language(text):
    global LDmodel
    LDfile=open('LDmodel2.pck1','rb')
    LDmodel= pickle.load(LDfile)
    LDfile.close()

    text=" ".join(text.split())
    text=text.lower()
    text=re.sub(r"\d+","",text)
    text=text.translate(table)
    prediction=LDmodel.predict([text])
    prob=LDmodel.predict_proba([text])
    return prediction[0]

language(input("Enter the word:"))
