import numpy as np
import math
import matplotlib.pyplot as plt
# data - 1000 * 10
# 

def logistic(a):
    return (1/(1+math.exp(-a)))

def relu(a):
    if a>0:
        return a
    else:
        return 0

def relu_derivative(a):
    if a>0:
        return 1
    else:
        return 0

def linear(a):
    return a

def logistic_derivative(a):
    return a*(1-a)

def tan_hyperbolic(a):
    None

def get_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    np.random.seed(42)
    np.random.shuffle(data)
    return data

class Neural_Network():
    data = None
    layers = None
    weights = None
    del_weights = None
    gradients = None
    activation = None
    activation_func = None
    biases = None
    MAX_EPOCH = 1
    learning_rate = 0.01

    def split_data(self):
        data_x = self.data[:,:-1]
        data_y = self.data[:,-1]
        data_y = np.reshape(data_y, (data_y.shape[0], 1))

        self.train_x = data_x[:int(len(data_x)*0.6),:]
        self.val_x = data_x[int(len(data_x)*0.6):int(len(data_x)*0.8),:]
        self.test_x = data_x[int(len(data_x)*0.8):,:]

        self.train_y = data_y[:int(len(data_y)*0.6),:]
        self.val_y = data_y[int(len(data_y)*0.6):int(len(data_y)*0.8),:]
        self.test_y = data_y[int(len(data_y)*0.8):,:]


    def predict(self, samples):
        predictions = []
        for i in range(len(samples)):
            self.activation[0] = np.reshape(samples[i], (samples[i].shape[0],1))
            self.forward_pass()
            predictions.append( self.activation[len(self.layers)][0][0])
        return predictions

    def construct(self, data, layers, activation_func):
        self.data = data
        input_dimen = data.shape[1]-1
        self.split_data()
        self.layers = layers
        self.weights = []
        self.del_weights = []
        self.gradients = []
        self.biases = []
        self.activation_func = activation_func
        self.activation = [np.zeros((input_dimen, 1))]
        for i in range(len(layers)):
            self.activation.append(np.zeros((layers[i], 1)))
            if i==0:
                self.gradients.append(np.random.random((input_dimen, layers[i])))
                self.weights.append(np.random.random((input_dimen, layers[i])))
                self.del_weights.append(np.random.random((input_dimen, layers[i])))
                self.biases.append(np.random.random((layers[i],1)))
            else:
                self.gradients.append(np.random.random((layers[i-1], layers[i])))
                self.weights.append(np.random.random((layers[i-1], layers[i])))
                self.del_weights.append(np.random.random((layers[i-1], layers[i])))
                self.biases.append(np.random.random((layers[i], 1)))

    def train(self):
        self.history=[]
        for i in range (self.MAX_EPOCH):
            percentage_done = i*100//self.MAX_EPOCH
            print('o'*(percentage_done) + '-'*(100-percentage_done), ": "+str(percentage_done)+"%",end='\r')

            av_error = self.epoch()
            self.history.append((i+1, av_error))
            if(len(self.history)>1 and self.history[i][1]==self.history[i-1][1]):
                break

        percentage_done = 100
        print('o'*(percentage_done) + '-'*(100-percentage_done), ": "+str(percentage_done)+"%",end='\r')
        print()
        

    def epoch(self):
        total_error = 0
        for i in range(len(self.train_x)):
            self.activation[0] = np.reshape(self.train_x[i,:], (self.train_x[i,:].shape[0],1))
            self.true_value = np.reshape(self.train_y[i,:], (self.train_y[i,:].shape[0],1))
            self.forward_pass()
            instaneous_error = 0.5*np.sum((self.true_value - self.activation[-1])**2)
            total_error += instaneous_error
            self.backward_pass()
        av_error = total_error/len(self.train_x)
        return av_error

    def forward_pass(self):
        for i in range(len(self.layers)):
            self.forward_propagation_layer(i)
            
    
    def forward_propagation_layer(self, current_layer):
        self.activation[current_layer+1] = np.matmul(self.weights[current_layer].T, self.activation[current_layer])
        self.activation[current_layer+1] += self.biases[current_layer]
        for i in range(len(self.activation[current_layer+1])):
            # print(self.activation[current_layer+1][i],end=' ')
            self.activation[current_layer+1][i] = self.activation_func[current_layer](self.activation[current_layer+1][i])
            # print(self.activation[current_layer+1][i])

    def backward_pass(self):
        for i in range(len(self.layers)-1, -1, -1):
            self.backward_propagation_layer(i)

    def backward_propagation_layer(self, current_layer):
        if current_layer == len(self.layers)-1:
            self.kronicker_delta = (self.true_value - self.activation[current_layer+1])    
        else:
            self.kronicker_delta = np.matmul(self.cache_weights, self.kronicker_delta)

        if self.activation_func[current_layer] == logistic:
            self.kronicker_delta *= logistic_derivative(self.activation[current_layer+1])
        elif self.activation_func[current_layer] == linear:
            self.kronicker_delta *= 1
        elif self.activation_func == tan_hyperbolic:
            self.kronicker_delta *= tan_hyperbolic_derivative(self.activation[current_layer+1])
        elif self.activation_func == relu:
            self.kronicker_delta *= relu_derivative(self.activation[current_layer+1])


        self.del_weights[current_layer] = self.learning_rate * np.matmul(self.activation[current_layer] ,self.kronicker_delta.T)
        self.cache_weights = self.weights[current_layer]
        self.weights[current_layer] += self.del_weights[current_layer]
        self.biases[current_layer] += self.learning_rate * self.kronicker_delta
    
    def training_prediction_graph(self):
        train_pred = self.predict(self.train_x)
        plt.scatter(self.train_x, self.train_y, label="actual")
        plt.scatter(self.train_x, train_pred, label="predictions")
        plt.legend()
        plt.title("Training Dataset")
        plt.show()
         
def main():
    data = get_data('./Group02/Regression/UnivariateData/2.csv')
    # normalise
    means = data.mean(axis=0)
    std = data.std(axis=0)
    new_matrix = data - means / std
    model = Neural_Network()
    model.construct(new_matrix, [5,5,1], [logistic,logistic,linear])
    model.learning_rate = 0.01
    model.MAX_EPOCH = 1500
    model.train()
    plt.plot([model.history[i][0] for i in range(len(model.history))], [model.history[i][1] for i in range(len(model.history))])
    plt.show()
    model.training_prediction_graph()

main()