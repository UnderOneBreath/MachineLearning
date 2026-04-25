import numpy as np

from lab02_old import super_simple_separable_through_origin, super_simple_separable, xor, xor_more
from lab02_old import data1, labels1, data2, labels2
from lab02_old import big_data, big_data_labels, gen_big_data

def perceptron(data, labels, tau=100):
    d, n = data.shape
    th = np.zeros((d, 1))
    th0 = np.zeros((1, 1))
    for t in range(tau):
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
    # print("1: ", temp)
    return temp
def positive(data, th, th0):
    sd = signed_dist(data, th, th0)
    # print("2: ", np.sign(sd))
    return np.sign(sd)
def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels) / data.shape[1]


def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0 = learner(data_train, labels_train)
    # print(score(data_test, labels_test, th, th0))
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
        # print("data_train.shape = ", data_train.shape)
        D_minus_j_labels = np.concatenate([D_i_labels[i] for i in range(k) if i != j], axis=1)
        # print("labels_train.shape = ", labels_train.shape)
        score += eval_classifier(learner, D_minus_j_data, D_minus_j_labels, data_test, labels_test)
    return score / k


if __name__ == "__main__":
    # print(perceptron(data1, labels1))

    # print(data1[:, 0])
    # print(data1[:, 2:2+1])
    # print(len(data1[0]), len(labels1[0]))
    # print(data1[:, :])
    # print(data1[:, 1].reshape(2, 1))

    print("4")
    for n in [super_simple_separable_through_origin, super_simple_separable]:
        data, labels = n()
        th, th0 = perceptron(data, labels)
        print("th: \n", th)
        print("th0: \n", th0)
    print()

    print("5")
    for n in [super_simple_separable_through_origin, super_simple_separable]:
        data, labels = n()
        th, th0 = averaged_perceptron(data, labels)
        print("th: \n", th)
        print("th0: \n", th0)
    print()

    print("6")
    print("perceptron dataset 1 -> dataset 2: ", eval_classifier(perceptron, data1, labels1, data2, labels2))
    print("perceptron dataset 2 -> dataset 1: ", eval_classifier(perceptron, data2, labels2, data1, labels1))
    print("averaged_perceptron dataset 1 -> dataset 2: ", eval_classifier(averaged_perceptron, data1, labels1, data2, labels2))
    print("averaged_perceptron dataset 2 -> dataset 1: ", eval_classifier(averaged_perceptron, data2, labels2, data1, labels1))
    print()

    print("7")
    gen_big_data = gen_big_data()
    print("perceptron: ", eval_learning_alg(perceptron, gen_big_data, 10, 10, 100))
    print("averaged_perceptron: ", eval_learning_alg(averaged_perceptron, gen_big_data, 10, 10, 100))
    print()

    print("8")
    print("perceptron k = 20: ", xval_learning_alg(perceptron, big_data, big_data_labels, 20))
    print("averaged_perceptron k = 20: ", xval_learning_alg(averaged_perceptron, big_data, big_data_labels, 20))
