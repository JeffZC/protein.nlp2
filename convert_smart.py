import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
from string import digits
import os


whole_ss3 = ""

 
def extract(df):
    temp = df["SS"].to_json()
    temp = temp.replace("\":\"", "")
    temp = temp.replace("\",\"", "")
    table = temp.maketrans('', '', digits)
    temp = temp.translate(table)
    temp = temp[2:-2]
    return temp

count = 0
path = "ss3/*.ss3"

if not os.path.exists("ss3-txt"):
    os.makedirs("ss3-txt")
    
for fpath in glob.glob(path):
    data = pd.read_table(fpath)
    ss3_txt = extract(data)
    #print(ss3_txt)
    wpath = "ss3-txt"+fpath[3:]+".txt"
    #print(wpath)
    f = open(wpath, "w")
    f.write(ss3_txt)
    f.close()
    whole_ss3 = whole_ss3 + extract(data) + "\n"


f = open("token.numpy", "w")

df = pd.DataFrame([], columns=list('AB'))

def process(tar):
    count = whole_ss3.count(tar)
    f = open("token.numpy", "a")
    f.write(tar)
    f.write(str(count))
    return count



for i in "CHE":
    for j in "CHE":
        target = i + j
        count = process(target)
        df= pd.concat([df, pd.DataFrame.from_records([{'A':target,'B':count}])])


for i in "CHE":
    for j in "CHE":
        for k in "CHE":
            target = i + j + k
            count = process(target)
            df= pd.concat([df, pd.DataFrame.from_records([{'A':target,'B':count}])])



for i in "CHE":
    for j in "CHE":
        for k in "CHE":
            for l in "CHE":
                target = i + j + k + l
                count = process(target)
                df= pd.concat([df, pd.DataFrame.from_records([{'A':target,'B':count}])])



for i in "CHE":
    for j in "CHE":
        for k in "CHE":
            for l in "CHE":
                for m in "CHE":
                    target = i + j + k + l + m
                    count = process(target)
                    df= pd.concat([df, pd.DataFrame.from_records([{'A':target,'B':count}])])



for i in "CHE":
    for j in "CHE":
        for k in "CHE":
            for l in "CHE":
                for m in "CHE":
                    for n in "CHE":
                        target = i + j + k + l + m + n
                        count = process(target)
                        df= pd.concat([df, pd.DataFrame.from_records([{'A':target,'B':count}])])



for i in "CHE":
    for j in "CHE":
        for k in "CHE":
            for l in "CHE":
                for m in "CHE":
                    for n in "CHE":
                        for o in "CHE":
                            target = i + j + k + l + m + n + o
                            count = process(target)
                            df= pd.concat([df, pd.DataFrame.from_records([{'A':target,'B':count}])])



for i in "CHE":
    for j in "CHE":
        for k in "CHE":
            for l in "CHE":
                for m in "CHE":
                    for n in "CHE":
                        for o in "CHE":
                            for p in "CHE":
                                target = i + j + k + l + m + n + o + p
                                count = process(target)
                                df= pd.concat([df, pd.DataFrame.from_records([{'A':target,'B':count}])])


f.close()
df.sort_values(by=['B'])
df.to_csv('./token.csv')

#print(data.sum())
#vectorizer = TfidfVectorizer(input = result, ngram_range = (2,6))

#X = vectorizer.fit_transform(corpus)
#func = vectorizer.build_tokenizer()
#print(vectorizer)
