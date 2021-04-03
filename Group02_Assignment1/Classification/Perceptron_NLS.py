import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

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
def calculate_accuracy(true , predicted):
    hits = 0
    con_matrix = [[0,0,0],[0,0,0],[0,0,0]]
    true = true.to_numpy()
    for i in range(len(true)):
        con_matrix[true[i]-1][predicted[i]-1] = con_matrix[true[i]-1][predicted[i]-1] +1
        if(true[i]==predicted[i]):
            hits = hits + 1
    return hits/len(true) , con_matrix
    
def splitter(filename1,filename2,class_name1 , class_name2, header):
    dataset1 = spit(filename1)
    dataset1['2'] = 0
    dataset1['3'] = class_name1

    dataset2 = spit(filename2)
    dataset2['2'] = 1
    dataset2['3'] = class_name2
    dataset = pd.concat([dataset1,dataset2])
    dataset = dataset.sample(frac = 1, random_state = 42).reset_index(drop = True)
    n = len(dataset)
    
    train = dataset[:int(0.6*n)]
    val = dataset[int(0.6*n):int(0.8*n)]
    test = dataset[int(0.8*n):]

    train_t = train['3']
    test_t = test['3']
    val_t = val['3']
    train = train.drop('3', 1)
    test = test.drop('3', 1)
    val = val.drop('3', 1)

    return train, val, test , train_t , val_t , test_t

# here input is augmented vector
class Perceptron:

    weights = None
    train = None
    total_error = None
    history=None
    val_loss_avg=None
    test_loss_avg = None

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def activation(self, a):
        return (1 / (1 + math.exp(-1*a)))

    def predict(self, dataset):
        #dataset = dataset.to_numpy()
        dataset_x = dataset[:,:-1]
        dataset_y = dataset[:,-1]
        dataset_x = np.concatenate((np.array([1 for i in range(len(dataset_x))]).reshape((len(dataset_x),1)), dataset_x), axis =1) # augmented set N x d+1
        #print("Hello")
        #print(dataset_x)
        predictions = self.regress(dataset_x)
        predictions = predictions.reshape((len(dataset),1))
        output = np.array([(1 if i>0.5 else 0) for i in predictions])
        return output

    def regress(self, sample):
        return np.matmul(self.weights.T , sample.T)

    # init params accroding to dataset
    def fit(self, data, val_data, test_data):
        print("Fitting the training dataset")
        data = data.to_numpy()
        test_data = test_data.to_numpy()
        val_data = val_data.to_numpy()
        actual = data[:,-1]
        test_actual = test_data[:,-1]
        val_actual = val_data[:,-1]
        train = data[:,:-1]
        test = test_data[:,:-1]
        val = val_data[:,:-1]
        print("Training data shape:",train.shape)
        print("Validation data shape:",val.shape)
        print("Test data shape:",test.shape)
        train = np.concatenate((np.array([1 for i in range(len(train))]).reshape((len(train),1)), train), axis =1) # augmented set N x d+1
        test = np.concatenate((np.array([1 for i in range(len(test))]).reshape((len(test),1)), test), axis =1) # augmented set N x d+1
        val = np.concatenate((np.array([1 for i in range(len(val))]).reshape((len(val),1)), val), axis =1) # augmented set N x d+1
        self.train = train
        self.test = test
        self.val = val
        self.actual = actual
        self.val_actual = val_actual
        self.test_actual = test_actual
        self.weights = np.random.uniform(low=0, high=0.001, size=(train.shape[1],1)) # N x 1 vector
        print("weights shape:",self.weights.shape)

    # to learn from the training dataset
    def learn(self):
        MAX_EPOCHS = 500 # Just to avoid computer hangs in case computations go out of bounds
        self.total_error = 1000000000
        self.flag_max_epoch_reached = True
        self.history = []
        self.val_loss_avg = []
        self.test_loss_avg = []
        for i in range(MAX_EPOCHS):
            current_error = 0

            # for training losses
            for n in range(len(self.train)):
                a = self.regress(self.train[n])
                s = self.activation(a)
                inst_error = 0.5 *((self.actual[n] - s)**2)
                current_error += inst_error
                del_weights = self.learning_rate*(self.actual[n] - s)*self.train[n]
                self.weights = self.weights + del_weights.T.reshape((self.weights.shape))
            self.history.append(current_error/len(self.train))

            # for validation data losses
            val_error = 0
            for n in range(len(self.val)):
                a = self.regress(self.val[n])
                s = self.activation(a)
                inst_error = 0.5* ((self.val_actual[n] - s)**2)
                val_error+= inst_error
            self.val_loss_avg.append(val_error/len(self.val))

            # if((self.total_error - current_error) / self.total_error < 0.0001):
            #     self.flag_max_epoch_reached = False
            #     #break
            # self.total_error = current_error

            #for test data
            test_error = 0
            #print("here",self.test)
            
            for n in range(len(self.test)):
                a = self.regress(self.test[n])
                s = self.activation(a)
                #print(self.test[n])
                inst_error = 0.5* ((self.test_actual[n] - s)**2)
                test_error+= inst_error
           # print("Helloj,",test_error/len(self.test))
            self.test_loss_avg.append(test_error/len(self.test))

            # if((self.total_error - current_error) / self.total_error < 0.0001):
            #     self.flag_max_epoch_reached = False
            #     #break
            # self.total_error = current_error
    
    # to predict all the values of the test dataset
    def evaluate(self, dataset):
        # test rms error
        predictions = self.predict(dataset)
        dataset = dataset.to_numpy()
        dataset_x = dataset[:,:-1]
        dataset_y = dataset[:,-1]
        rms_error = (1/len(dataset))*(((predictions - dataset_y)**2).reshape(-1).sum())
        rms_error = rms_error**0.5
        return rms_error

class multi_classifier:
    model12 = Perceptron(learning_rate=0.001)
    model23 = Perceptron(learning_rate=0.001)
    model31 = Perceptron(learning_rate=0.001)
    train_data12 = None
    val_data12 = None
    test_data12 = None
    
    train_data23 = None
    val_data23 = None
    test_data23 = None
    
    train_data31 = None
    val_data31 = None
    test_data31 = None

    filename1 = None
    filename2 = None
    filename3 = None
    
    history = []
    val_loss_avg =  []
    test_loss_avg = []
    def feed_data(self,filename1,filename2,filename3):
        self.train_data12, self.val_data12, self.test_data12,a,b,c = splitter(filename1,filename2,1,2, None)
        self.train_data23, self.val_data23, self.test_data23,a,b,c = splitter(filename2,filename3,'2','3', None)
        self.train_data31, self.val_data31, self.test_data31,a,b,c = splitter(filename3,filename1,'3','1', None)
        
        
    def train(self):

        self.model12.fit(self.train_data12, self.val_data12, self.test_data12)
        self.model12.learn()
        self.model23.fit(self.train_data23, self.val_data23, self.test_data23)
        self.model23.learn()
        self.model31.fit(self.train_data31, self.val_data31, self.test_data31)
        self.model31.learn()
        for i in range(len(self.model12.history)):
            self.history.append((self.model12.history[i] + self.model23.history[i] + self.model31.history[i])/3)
            self.val_loss_avg.append((self.model12.val_loss_avg[i] + self.model23.val_loss_avg[i] + self.model31.val_loss_avg[i])/3)
            self.test_loss_avg.append((self.model12.test_loss_avg[i] + self.model23.test_loss_avg[i] + self.model31.test_loss_avg[i])/3)
    def predict(self , samples):
        predicted_classes = []
        samples = samples.to_numpy()
        #print(samples)
        samples = samples.reshape((len(samples),1,3))
        
        for sample in samples:
            votes = {1:0,2:0,3:0}
            #print("sample is",sample)
            if(self.model12.predict(sample)[0]==0):
                votes[1] = votes[1] +1
            else:
                votes[2] = votes[2] + 1
            if(self.model23.predict(sample)[0]==0):
                votes[2] = votes[2] + 1
            else:
                votes[3] = votes[3] + 1
            if(self.model31.predict(sample)[0]==0):
                
                votes[3] = votes[3] + 1
            else:
                votes[1] = votes[1] + 1



            predicted_class = max(votes, key= lambda x: votes[x])
            predicted_classes.append(predicted_class)
        return predicted_classes

filename1 = "Group02/Classification/Class1.txt"
filename2 = "Group02/Classification/Class2.txt"
filename3 = "Group02/Classification/Class3.txt"
train_data12, val_data12, test_data12 , train_data12_true , val_data12_true ,test_data12_true  = splitter(filename1,filename2,1,2 ,None)
train_data23, val_data23, test_data23 , train_data23_true , val_data23_true ,test_data23_true = splitter(filename2,filename3,2,3, None)
train_data31, val_data31, test_data31 , train_data31_true , val_data31_true ,test_data31_true= splitter(filename3,filename1,3,1, None)

#print(test_data31)
model = multi_classifier()
model.feed_data(filename1,filename2,filename3)
model.train()
total_train = pd.concat([train_data12 , train_data23 , train_data31])
total_test = pd.concat([test_data12 , test_data23 , test_data31])
total_train_true = pd.concat([train_data12_true , train_data23_true , train_data31_true])
total_test_true = pd.concat([test_data12_true , test_data23_true , test_data31_true])
print("The accuracy is",calculate_accuracy(total_test_true , model.predict(total_test)))

##plot for error AND loss
# print(model.test_loss_avg)
# print(model.history)
# plt.plot([int(i) for i in range(1,len(model.history) + 1)] , model.history , label="train_loss")
# plt.plot([int(i) for i in range(1,len(model.history) + 1)], model.val_loss_avg, label ="Validation Loss")
# plt.plot([int(i) for i in range(1,len(model.history) + 1)], model.test_loss_avg, label ="test Loss")
# plt.xlabel("Number of Epochs")
# plt.ylabel("Average Error")
# plt.legend()
# plt.savefig("Epoch graph for NLS")
# plt.show()

### plotting the desicion boundary

# xx = np.linspace(-15,15,num=200)
# yy = np.linspace(-15,15,num=200)
# class1 = []
# class2 = []
# class3 = []
# for x in xx:
#     for y in yy:
#         sample = pd.DataFrame(columns=['1','2','3']) 
#         sample.loc[0] = [x,y,1]
#         #print(sample)
#         if(model.predict(sample)[0]==1):
#             class1.append(sample)
#         elif(model.predict(sample)[0]==2):
#             class2.append(sample)
#         else:
#             class3.append(sample)
# class3 = np.array(class3)
# class2 = np.array(class2)
# class1 = np.array(class1)

# plt.scatter(class1[:,:,0],class1[:,:,1] , label="class 1")
# plt.scatter(class2[:,:,0] , class2[:,:,1] , label="class 2")
# plt.scatter(class3[:,:,0] , class3[:,:,1] , label="class 3")
# plt.legend()
# plt.savefig("DB for NLS")
# plt.show()
    


