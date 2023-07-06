import numpy as np


def sigmoid(x):
    # Activation function, f(x) = 1 / (1+ e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # derivative of the activation function, f'(x) = f(x) * (1-f(x))
    fx = sigmoid(x)
    return (fx * (1 - fx))


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class Neuron:
    def __init__(self, weight1, weight2, bias):
        self.weight1 = weight1
        self.weight2 = weight2
        self.bias = bias

    def neuronfeedforward(self, inputs):
        return sigmoid((self.weight1 * inputs[0]) + (self.weight2 * inputs[1]) + self.bias)


class NeuralNetwork:
    def __init__(self):

        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

        self.h1 = Neuron(self.w1, self.w2, self.b1)
        self.h2 = Neuron(self.w3, self.w4, self.b2)
        self.o1 = Neuron(self.w5, self.w6, self.b3)

    def networkfeedforward(self, inputs):
        out_h1 = self.h1.neuronfeedforward(inputs)
        out_h2 = self.h2.neuronfeedforward(inputs)

        out_o1 = self.o1.neuronfeedforward(np.array([out_h1, out_h2]))

        return out_o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000  # Number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Do a feedforward
                sum_h1 = (self.w1 * x[0]) + (self.w2 * x[1]) + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = (self.w3 * x[0]) + (self.w4 * x[1]) + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = (self.w5 * x[0]) + (self.w6 * x[1]) + self.b3
                o1 = sigmoid(sum_o1)

                y_pred = o1

                # Calculate partial derivatives
                d_L_d_ypred = -2 * (y_true-y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                #Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.networkfeedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                # print("Epoch %d loss: %.3f" % (epoch, loss))
                print(y_preds)


data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = NeuralNetwork()
network.train(data, all_y_trues)