import numpy as np


# Датасеты

def super_simple_separable_through_origin():
    X = np.array([[2, 3, 9, 12],
                  [5, 1, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def xor():
    X = np.array([[1, 2, 1, 2],
                  [1, 2, 2, 1]])
    y = np.array([[1, 1, -1, -1]])
    return X, y

def xor_more():
    X = np.array([[1, 2, 1, 2, 2, 4, 1, 3],
                  [1, 2, 2, 1, 3, 1, 3, 3]])
    y = np.array([[1, 1, -1, -1, 1, 1, -1, -1]])
    return X, y

def dataset_1():
    X = np.array(
        [[ -2.97797707,  2.84547604,  3.60537239, -1.72914799, -2.51139524, 3.10363716, 2.13434789, 1.61328413, 2.10491257, -3.87099125, 3.69972003, -0.23572183, -4.19729119, -3.51229538, -1.75975746, -4.93242615, 2.16880073, -4.34923279, -0.76154262, 3.04879591, -4.70503877,  0.25768309,  2.87336016,  3.11875861, -1.58542576, -1.00326657, 3.62331703, -4.97864369, -3.31037331, -1.16371314 ],
        [ 0.99951218, -3.69531043, -4.65329654, 2.01907382, 0.31689211, 2.4843758, -3.47935105, -4.31857472, -0.11863976,  0.34441625, 0.77851176, 1.6403079, -0.57558913, -3.62293005, -2.9638734, -2.80071438, 2.82523704, 2.07860509, 0.23992709, 4.790368, -2.33037832, 2.28365246, -1.27955206, -0.16325247, 2.75740801, 4.48727808, 1.6663558, 2.34395397, 1.45874837, -4.80999977 ]])
    y = np.array([[-1., -1., -1., -1., -1., -1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1., -1.,  1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.]])
    return X, y

def dataset_2():
    X = np.array(
        [[ -2.97797707, 2.84547604, 3.60537239, -1.72914799, -2.51139524, 3.10363716, 2.13434789, 1.61328413, 2.10491257, -3.87099125, 3.69972003, -0.23572183, -4.19729119, -3.51229538, -1.75975746, -4.93242615, 2.16880073, -4.34923279, -0.76154262, 3.04879591, -4.70503877, 0.25768309, 2.87336016, 3.11875861, -1.58542576, -1.00326657, 3.62331703, -4.97864369, -3.31037331, -1.16371314],
        [ 0.99951218, -3.69531043, -4.65329654,  2.01907382,  0.31689211, 2.4843758, -3.47935105, -4.31857472, -0.11863976, 0.34441625, 0.77851176, 1.6403079, -0.57558913, -3.62293005, -2.9638734, -2.80071438, 2.82523704, 2.07860509, 0.23992709, 4.790368, -2.33037832, 2.28365246, -1.27955206, -0.16325247, 2.75740801, 4.48727808, 1.6663558, 2.34395397, 1.45874837, -4.80999977]])
    y = np.array([[ -1., -1., 1., 1., -1., -1., -1., 1., 1., 1., -1., 1., 1., -1., 1., 1., 1., -1., -1., -1., 1., -1., 1., -1., 1., -1., -1., 1., 1., 1.]])
    return X, y


    
# Ваше решение идёт тут

def perceptron(data, labels, tau=100):
    d, n = data.shape
    th = np.zeros((d, 1))
    th0 = np.zeros((1, 1))
    for t in range(tau):
        for i in range(n):
            x = data[:, i:i+1]
            y = labels[:, i:i+1]
            if y * (th.T @ x + th0) <= 0:
                th = th + y * x
                th0 = th0 + y
    return th, th0
      
def averaged_perceptron(data, labels, params={}, hook=None):
    T = params.get('T', 100)
    d, n = data.shape
    th = np.zeros((d, 1))
    th0 = np.zeros((1, 1))
    ths = np.zeros((d, 1))
    th0s = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:, i:i+1]
            y = labels[:, i:i+1]
            if y * (th.T @ x + th0) <= 0:
                th = th + y * x
                th0 = th0 + y
            ths = ths + th
            th0s = th0s + th0
    return ths / (n * T), th0s / (n * T)


def signed_dist(data, th, th0):
    temp = (th.T @ data + th0) / np.linalg.norm(th)
    return temp
def positive(data, th, th0):
    sd = signed_dist(data, th, th0)
    return np.sign(sd)
def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels) / data.shape[1]


def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0 = learner(data_train, labels_train)
    return score(data_test, labels_test, th, th0)


def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    total = 0.0
    for i in range(it):
        data_train, labels_train = data_gen(n_train)
        data_test, labels_test = data_gen(n_test)
        total = total + eval_classifier(learner, data_train, labels_train, data_test, labels_test)
    return total / it

def xval_learning_alg(learner, data, labels, k):
    D_i_data = np.array_split(data, k, axis=1)
    D_i_labels = np.array_split(labels, k, axis=1)
    score = 0.0
    for j in range(k):
        data_test = D_i_data[j]
        labels_test = D_i_labels[j]
        D_minus_j_data = np.concatenate([D_i_data[i] for i in range(k) if i != j], axis=1)
        D_minus_j_labels = np.concatenate([D_i_labels[i] for i in range(k) if i != j], axis=1)
        score += eval_classifier(learner, D_minus_j_data, D_minus_j_labels, data_test, labels_test)
    return score / k