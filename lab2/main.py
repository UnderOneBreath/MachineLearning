import numpy as np
from lab02 import data1, data2, big_data
from lab02 import labels1, labels2, big_data_labels

from lab02 import test_perceptron

# def check_size(data, labels):
    
    

def perceptron(data, labels, params={}, hook=None):
    T = params.get('T', 5)
    d, n = data.shape
    th = np.zeros((d, 1))
    th0 = np.zeros((1, 1))

    for t in range(T):
        for i in range(n):
            x = data[:, i:i+1]
            y = labels[:, i:i+1]
            # print(x, y)
            # print("th: ", th.T)
            # print("my if: ", y * (th.T @ x + th0))
            if y * (th.T @ x + th0) <= 0:
                th = th + y * x
                th0 = th0 + y
                # print("th: ", th.T, "th0: ", th0)
            # print()
    return th, th0

if __name__ == "__main__":
    # print(perceptron(data1, labels1))
    test_perceptron(perceptron)

    # print(data1[:, 0])
    # print(data1[:, 2:2+1])
    # print(len(data1[0]), len(labels1[0]))
    # print(data1[:, :])
    # print(data1[:, 1].reshape(2, 1))