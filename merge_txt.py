import os
import glob

path = "ss3-txt/*"

for fpath in glob.glob(path):
    f1 = open(fpath, "r")
    f2 = open("train_ds", "a")
    content = f1.readline()
    f2.write(content)
    f2.write("\n")
    f1.close()
    f2.close()
