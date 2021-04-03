import matplotlib.pyplot as plt
import pandas as pd
def spit(filename):
    f = open(filename)
    df = pd.DataFrame(columns=['0', '1'])
    i=0
    for line in f:
        l = line.split()
        l = [float(i) for i in l]
        df.loc[i] = l
    ##print(line)
        i=i+1
    return df

filename1 = "Group02/Classification/LS_Group02/Class1.txt"
filename2 = "Group02/Classification/LS_Group02/Class2.txt"
filename3 = "Group02/Classification/LS_Group02/Class3.txt"

df = spit(filename1)
df2 = spit(filename2)
df3 = spit(filename3)
plt.scatter(df['0'],df['1'],label="class1")
plt.scatter(df2['0'] , df2['1'] , label="class2")
plt.scatter(df3['0'] , df3['1'] , label="class3")
plt.legend()
plt.savefig("data for NLS")
plt.show()
