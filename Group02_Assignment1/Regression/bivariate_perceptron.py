import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def splitter(filename, header):
    dataset = pd.read_csv(filename, header = header)
    dataset = dataset.sample(frac = 1, random_state = 42).reset_index(drop = True)
    n = len(dataset)
    train = dataset[:int(0.6*n)]
    val = dataset[int(0.6*n):int(0.8*n)]
    test = dataset[int(0.8*n):]
    return train, val, test

# here input is augmented vector
class Perceptron:

    weights = None
    train = None
    total_error = None
    history=None
    val_loss_avg=None

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def activation(self, a):
        return a

    def predict(self, dataset):
        dataset = dataset.to_numpy()
        dataset_x = dataset[:,:-1]
        dataset_y = dataset[:,-1]
        dataset_x = np.concatenate((np.array([1 for i in range(len(dataset_x))]).reshape((len(dataset_x),1)), dataset_x), axis =1) # augmented set N x d+1
        predictions = self.regress(dataset_x)
        return predictions.reshape((len(dataset),1))

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
        MAX_EPOCHS = 10000 # Just to avoid computer hangs in case computations go out of bounds
        self.total_error = 1000000000
        self.flag_max_epoch_reached = True
        self.history = []
        self.val_loss_avg = []
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

            if((self.total_error - current_error) / self.total_error < 0.0001):
                self.flag_max_epoch_reached = False
                break
            self.total_error = current_error
    
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
        

def main():
    filename = "BivariateData/2.csv"
    graphs_dir = "graphs_bivariate_perceptron/"
    train_data, val_data, test_data = splitter(filename, None)
    model = Perceptron(learning_rate=0.001)
    model.fit(train_data, val_data, test_data)
    model.learn()
    predictions = model.predict(test_data)
    total_epochs = len(model.history)
    h = model.history
    vh = model.val_loss_avg

    # plottin the epochs vs losses curve
    plt.plot([i for i in range(1,total_epochs+1)], h, label = "Training Loss" )
    plt.plot([i for i in range(1,total_epochs+1)], vh, label = "Validation Loss" )
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Error")
    plt.grid()
    plt.legend()
    plt.savefig(graphs_dir+'loss_curve.png')
    plt.close()

    # RMS values
    print()
    test_rms = model.evaluate(test_data)
    print("Test RMS error:",test_rms)
    train_rms = model.evaluate(train_data)
    print("Training RMS error:",train_rms)
    val_rms = model.evaluate(val_data)
    print("validation RMS error:", val_rms)

    # plots of values actual vs x and predited vs x
    # on training
    train_predictions = model.predict(train_data)
    ax = plt.axes(projection='3d')
    ax.scatter3D(train_data[0], train_data[1], train_data[2], label='Actual')
    ax.scatter3D(train_data[0], train_data[1], train_predictions, label = "Predicted")
    plt.title("Training")
    ax.set_xlabel("Attr1")
    ax.set_ylabel("Attr2")
    ax.set_zlabel('Value')
    plt.legend()
    plt.savefig(graphs_dir+'dist_train.png')
    plt.close()

    # # on validation
    val_predictions = model.predict(val_data)
    ax = plt.axes(projection='3d')
    ax.scatter3D(val_data[0], val_data[1], val_data[2], label='Actual')
    ax.scatter3D(val_data[0], val_data[1], val_predictions, label = "Predicted")
    plt.title("Validation")
    ax.set_xlabel("Attr1")
    ax.set_ylabel("Attr2")
    ax.set_zlabel('Value')
    plt.legend()
    plt.savefig(graphs_dir+'dist_val.png')
    plt.close()

    # # on Testing
    test_predictions = model.predict(test_data)
    ax = plt.axes(projection='3d')
    ax.scatter3D(test_data[0], test_data[1], test_data[2], label='Actual')
    ax.scatter3D(test_data[0], test_data[1], test_predictions, label = "Predicted")
    plt.title("Testing")
    ax.set_xlabel("Attr1")
    ax.set_ylabel("Attr2")
    ax.set_zlabel('Value')
    plt.legend()
    plt.savefig(graphs_dir+'dist_test.png')
    plt.close()

    # Scatter Plots btw predictions and actual
    # on training
    train_predictions = model.predict(train_data)
    plt.scatter(train_data[1], train_predictions)
    plt.xlabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.title("Training Dataset")
    plt.savefig(graphs_dir+'actual_vs_pred_train.png')
    plt.close()

    # on validation
    val_predictions = model.predict(val_data)
    plt.scatter(val_data[1], val_predictions)
    plt.xlabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.title("Validation Dataset")
    plt.savefig(graphs_dir+'actual_vs_pred_val.png')
    plt.close()

    # on Testing
    test_predictions = model.predict(test_data)
    plt.scatter(test_data[1], test_predictions)
    plt.xlabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.title("Test Dataset")
    plt.savefig(graphs_dir+'actual_vs_pred_test.png')
    plt.close()

main()
