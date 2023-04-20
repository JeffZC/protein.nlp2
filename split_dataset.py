import numpy as np
import shutil

X_train = np.load('harder_task_train.npy')
X_test = np.load('harder_task_test.npy')
f = open('fasta_description.txt', 'r')
for i in range(0, 20290):
    fasta = f.readline()
    if "TATLIPO" in fasta:
        if i in X_train:
            shutil.copy("harder_task_dataset_txt/"+str(i)+".txt", "TATLIPO/train")
        elif i in X_test:
            shutil.copy("harder_task_dataset_txt/"+str(i)+".txt", "TATLIPO/test")
    elif "TAT" in fasta:
        if i in X_train:
            shutil.copy("harder_task_dataset_txt/"+str(i)+".txt", "TAT/train")
        elif i in X_test:
            shutil.copy("harder_task_dataset_txt/"+str(i)+".txt", "TAT/test")
    elif "NO_SP" in fasta:
        if i in X_train:
            shutil.copy("harder_task_dataset_txt/"+str(i)+".txt", "NO_SP/train")
        elif i in X_test:
            shutil.copy("harder_task_dataset_txt/"+str(i)+".txt", "NO_SP/test")
    elif "LIPO" in fasta:
        if i in X_train:
            shutil.copy("harder_task_dataset_txt/"+str(i)+".txt", "LIPO/train")
        elif i in X_test:
            shutil.copy("harder_task_dataset_txt/"+str(i)+".txt", "LIPO/test")
 
    
