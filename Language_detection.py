import pickle
import re
import string
"""table consist of punctuations"""
table = str.maketrans('', '', string.punctuation)

def language(text):
    """Opeaning the saved model LDmodel2.pck1 which was saved by Prediction.py.
    The model is saved in LDmodel"""
    global LDmodel
    LDfile=open('LDmodel2.pck1','rb')
    LDmodel= pickle.load(LDfile)
    LDfile.close()
    """allgning the text given by the user then change it to small letters and removing numeric values from the given text.
    Removing punctuations from the text and dredicting the model."""
    text=" ".join(text.split())
    text=text.lower()
    text=re.sub(r"\d+","",text)
    text=text.translate(table)
    prediction=LDmodel.predict([text])
    #prob=LDmodel.predict_proba([text])
    return prediction[0]

language(input("Enter the word:"))
