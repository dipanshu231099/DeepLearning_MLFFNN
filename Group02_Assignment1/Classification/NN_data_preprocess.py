import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def spit(filename , class_no):
    f = open(filename)
    df = pd.DataFrame(columns=['0', '1'])
    i=0

    for line in f:
        l = line.split()
        l = [float(i) for i in l]
        df.loc[i] = l
    ##print(line)
        i=i+1
        labels = []


    for i in range(len(df)):
        ll = [0]*3
        ll[class_no-1] = 1
        labels.append(ll)
    labels = np.array(labels)
    df = np.array(df)
    return df,labels


filename1 = "./Group02/Classification/Class1.txt"
filename2 = "./Group02/Classification/Class2.txt"
filename3 = "./Group02/Classification/Class3.txt"

df1,true_label1 = spit(filename1,1)
df2,true_label2 = spit(filename2,2)
df3,true_label3 = spit(filename3,3)
plt.scatter(df1[:,0],df1[:,1] , label="class 1")
plt.scatter(df2[:,0],df2[:,1] , label = "class 2")
plt.scatter(df3[:,0],df3[:,1] , label = "class 3")
plt.legend()
plt.savefig("DATA NLS")
plt.show()
print("hello")
print(df1[:0])




