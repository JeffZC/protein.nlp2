import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
from string import digits
import os

ini_path = "harder_task_dataset"
 
def extract(df):
    temp = df["SS"].to_json()
    temp = temp.replace("\":\"", "")
    temp = temp.replace("\",\"", "")
    table = temp.maketrans('', '', digits)
    temp = temp.translate(table)
    temp = temp[2:-2]
    return temp

path = ini_path+"/*.ss3"
des_dir = ini_path+"_txt"

if not os.path.exists(des_dir):
    os.makedirs(des_dir)
    
for i, fpath in enumerate(glob.glob(path)):
    data = pd.read_table(fpath)
    ss3_txt = extract(data)
    w_path = des_dir+"/"+str(i)+".txt"
    f = open(w_path, "w")
    f.write(ss3_txt)
    f.close()
