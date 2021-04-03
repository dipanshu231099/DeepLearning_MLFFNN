import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def spit(filename , class_no):
    f = open(filename)
    df = []
    i=0

    for line in f:
        l = line.split()
        l = [float(i) for i in l]
        df.append(l)
        i=i+1
    labels = []


    for i in range(len(df)):
        ll = [0]*3
        ll[class_no-1] = 1
        ll = np.array
        labels.append(ll)
    labels = np.array(labels)
    df = np.array(df)
    return df,labels


filename1 = "./Group02/Classification/LS_Group02/Class1.txt"
filename2 = "./Group02/Classification/LS_Group02/Class2.txt"
filename3 = "./Group02/Classification/LS_Group02/Class3.txt"

df1,a = spit(filename1,1)
df2,b = spit(filename2,2)
df3,c = spit(filename3,3)
df = np.concatenate((df1,df2,df3) , axis=0)
t = np.concatenate((a,b,c) , axis=0)
print("hello")
print(df1)
print(t)