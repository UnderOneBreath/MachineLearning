import numpy as np



# Вспомогательные функции

def rv(value_list):
    return np.array([value_list])

def cv(value_list):
    return np.transpose(rv(value_list))


# Функции для тестов

def f1(v):
    assert v.shape == (1, 1)
    x = float(v[0,0])
    return (2 * x + 3) ** 2

def df1(v):
    assert v.shape == (1, 1)
    x = float(v[0,0])
    return 2 * 2 * (2 * x + 3)

def f2(v):
    assert v.shape == (2, 1)
    x = float(v[0,0]); y = float(v[1,0])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y - 1.) ** 2

def df2(v):
    assert v.shape == (2, 1)
    x = float(v[0,0]); y = float(v[1,0])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + (-3. + x) * (-2. + x) * (3. + x) + (-3. + x) * (1. + x) * (3. + x) + (-2. + x) * (1. + x) * (3. + x) + 2 * (-1. + x + y), 2 * (-1. + x + y)])


def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    Y = np.array([[1, -1, 1, -1]])
    return X, Y

def separable_medium():
    X = np.array([[2, -1, 1, 1],
                  [-2, 2, 2, -1]])
    Y = np.array([[1, -1, 1, -1]])
    return X, Y

sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])


# Ваше решение идёт тут

# Задание 2

def grad_desc(f, df, x0, eta, T):
    x = x0
    for t in range(T):
        x = x - eta * df(x)
    return x

def num_grad(f, delta=0.001):
    def grad(x):
        n = x.shape[0]
        result = np.zeros_like(x, dtype=float)
        for i in range(n):
            delta_i = np.zeros_like(x, dtype=float)
            delta_i[i, 0] = delta
            result[i, 0] = (f(x + delta_i) - f(x - delta_i)) / (2 * delta)
        return result
    return grad

def num_grad_desc(f, x0, eta, T):
    return grad_desc(f, num_grad(f), x0, eta, T)


# Задание 3

def hinge(v):
    return np.where(v < 1, 1 - v, 0)

def hinge_loss(X, Y, th, th0):
    return hinge(Y * (th.T @ X + th0))

def svm_obj(X, Y, th, th0, lam):
    return np.mean(hinge_loss(X, Y, th, th0)) + lam * np.sum(th ** 2)


# Задание 4

def d_hinge(v):
    return np.where(v < 1, -1, 0).astype(float)

def d_hinge_loss_th(X, Y, th, th0):
    return X * (d_hinge(Y * (th.T @ X + th0)) * Y)

def d_hinge_loss_th0(X, Y, th, th0):
    return d_hinge(Y * (th.T @ X + th0)) * Y

def d_svm_obj_th(X, Y, th, th0, lam):
    return np.mean(d_hinge_loss_th(X, Y, th, th0), axis=1, keepdims=True) + 2 * lam * th

def d_svm_obj_th0(X, Y, th, th0, lam):
    return np.mean(d_hinge_loss_th0(X, Y, th, th0), axis=1, keepdims=True)

def svm_obj_grad(X, Y, th, th0, lam):
    return np.vstack([d_svm_obj_th(X, Y, th, th0, lam), d_svm_obj_th0(X, Y, th, th0, lam)])


# Задание 5

def svm_grad_desc(X, Y, lam, eta, T):
    d = X.shape[0]
    x0 = np.zeros((d + 1, 1))

    def f(x):
        th = x[:d, :]
        th0 = x[d:, :]
        return svm_obj(X, Y, th, th0, lam)

    def df(x):
        th = x[:d, :]
        th0 = x[d:, :]
        return svm_obj_grad(X, Y, th, th0, lam)

    return grad_desc(f, df, x0, eta, T - 1)