import numpy as np

# Далее, эти переменные определены ровно как на лекции:
# X - матрица d x n
# Y - матрица 1 x n
# th - матрица d x 1
# th0 - матрица 1 x 1


# Задание 2a

def lin_reg(X, th, th0):
    return (X.T @ th + th0).T

def square_loss(X, Y, th, th0):
    return (lin_reg(X, th, th0) - Y) ** 2

def mean_square_loss(X, Y, th, th0):
    return np.array([[np.mean(square_loss(X, Y, th, th0))]])


# Задание 2b

def d_lin_reg_th(X, th, th0):
    return X

def d_square_loss_th(X, Y, th, th0):
    return 2 * (lin_reg(X, th, th0) - Y) * d_lin_reg_th(X, th, th0)

def d_mean_square_loss_th(X, Y, th, th0):
    return np.mean(d_square_loss_th(X, Y, th, th0), axis=1, keepdims=True)


# Задание 2c

def d_lin_reg_th0(X, th, th0):
    return np.ones((1, X.shape[1]))

def d_square_loss_th0(X, Y, th, th0):
    return 2 * (lin_reg(X, th, th0) - Y)

def d_mean_square_loss_th0(X, Y, th, th0):
    return np.mean(d_square_loss_th0(X, Y, th, th0), axis=1, keepdims=True)


# Задание 2d

def ridge_obj(X, Y, th, th0, lam):
    return mean_square_loss(X, Y, th, th0) + lam * np.sum(th ** 2)

def d_ridge_obj_th(X, Y, th, th0, lam):
    return d_mean_square_loss_th(X, Y, th, th0)  + 2 * lam * th 

def d_ridge_obj_th0(X, Y, th, th0, lam):
    return d_mean_square_loss_th0(X, Y, th, th0)




# Задание 3

def stoc_grad_desc(X, Y, J, dJ, w0, eta, T):
    np.random.seed(0)
    w = w0.copy()
    n = X.shape[1]
    for i in range(T - 1):
        idx = np.random.randint(n)
        Xi = X[:, idx:idx+1]
        Yi = Y[:, idx:idx+1]
        w = w - eta * dJ(Xi, Yi, w)
    return w


# Датасет и соответствующие ей функции J и dJ для Задания 3

def downwards_line():
    X = np.array([[0.0, 0.1, 0.2, 0.3, 0.42, 0.52, 0.72, 0.78, 0.84, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]])
    Y = np.array([[0.4, 0.6, 1.2, 0.1, 0.22, -0.6, -1.5, -0.5, -0.5, 0.0]])
    return X, Y

X, Y = downwards_line()

def J(Xi, Yi, w):
    # перевод из формата (1-augmented X, Y, th) в (separated X, Y, th, th0) формат
    return float(ridge_obj(Xi[:-1,:], Yi, w[:-1,:], w[-1:,:], 0))

def dJ(Xi, Yi, w):
    grad_th = d_ridge_obj_th(Xi[:-1,:], Yi, w[:-1,:], w[-1:,:], 0)
    grad_th0 = d_ridge_obj_th0(Xi[:-1,:], Yi, w[:-1,:], w[-1:,:], 0)
    return np.vstack([grad_th, grad_th0])

if __name__ == "__main__":
    X, Y = downwards_line()
    
    params = [
        (np.array([[0.],[0.]]), 0.001, 10000),
        (np.array([[0.],[0.]]), 0.005, 1000),
        (np.array([[1.],[1.]]), 0.001, 5000),
        (np.array([[0.],[0.]]), 0.01, 500),
    ]
    
    for w0, eta, T in params:
        w = stoc_grad_desc(X, Y, J, dJ, w0, eta, T)
        print(f"w0={w0}, eta={eta}, T={T}")
        print(f"Result w = {w}\n")