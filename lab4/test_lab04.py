import numpy as np
from numpy.testing import assert_allclose
from lab04 import lin_reg, square_loss, mean_square_loss
from lab04 import d_lin_reg_th, d_square_loss_th, d_mean_square_loss_th
from lab04 import d_lin_reg_th0, d_square_loss_th0, d_mean_square_loss_th0
from lab04 import ridge_obj, d_ridge_obj_th, d_ridge_obj_th0
from lab04 import downwards_line, stoc_grad_desc, J, dJ

X = np.array([[1., 2., 3., 4.], [1., 1., 1., 1.]])
Y = np.array([[1., 2.2, 2.8, 4.1]])
th = np.array([[1.], [0.05]])
th0 = np.array([[2.]])

def test_lin_reg():
    ans = np.array([[1.05, 2.05, 3.05, 4.05]])
    assert_allclose(lin_reg(X, th, np.array([[ 0. ]])), ans)
    ans = np.array([[3.05, 4.05, 5.05, 6.05]])
    assert_allclose(lin_reg(X, th, th0), ans)

def test_square_loss():
    ans = np.array([[4.2025, 3.4224999999999985, 5.0625, 3.8025000000000007]])
    assert_allclose(square_loss(X, Y, th, th0), ans)

def test_mean_square_loss():
    ans = np.array([[4.1225]])
    assert_allclose(mean_square_loss(X, Y, th, th0), ans)

def test_d_lin_reg_th():
    ans = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]])
    assert_allclose(d_lin_reg_th(X, th, th0), ans)

def test_d_square_loss_th():
    ans = np.array([[4.1, 7.399999999999999, 13.5, 15.600000000000001], [4.1, 3.6999999999999993, 4.5, 3.9000000000000004]])
    assert_allclose(d_square_loss_th(X, Y, th, th0), ans)

def test_d_mean_sqaure_loss_th():
    ans = np.array([[10.15], [4.05]])
    assert_allclose(d_mean_square_loss_th(X, Y, th, th0), ans)

def test_d_lin_reg_th0():
    ans = np.array([[1.0, 1.0, 1.0, 1.0]])
    assert_allclose(d_lin_reg_th0(X, th, th0), ans)

def test_d_square_loss_th0():
    ans = np.array([[4.1, 3.6999999999999993, 4.5, 3.9000000000000004]])
    assert_allclose(d_square_loss_th0(X, Y, th, th0), ans)

def test_d_mean_square_loss_th0():
    ans = np.array([[4.05]])
    assert_allclose(d_mean_square_loss_th0(X, Y, th, th0), ans)


def test_ridge_obj():
    assert_allclose(ridge_obj(X, Y, th, th0, 0.0), np.array([[4.1225]]))
    assert_allclose(ridge_obj(X, Y, th, th0, 0.5), np.array([[4.623749999999999]]))
    assert_allclose(ridge_obj(X, Y, th, th0, 100.0), np.array([[104.37250000000002]]))

def test_d_ridge_obj_th():
    assert_allclose(d_ridge_obj_th(X, Y, th, th0, 0.0), np.array([[10.15], [4.05]]))
    assert_allclose(d_ridge_obj_th(X, Y, th, th0, 0.5), np.array([[11.15], [4.1]]))
    assert_allclose(d_ridge_obj_th(X, Y, th, th0, 100.0), np.array([[210.15], [14.05]]))

def test_d_ridge_obj_th0():
    assert_allclose(d_ridge_obj_th0(X, Y, th, th0, 0.0), np.array([[4.05]]))
    assert_allclose(d_ridge_obj_th0(X, Y, th, th0, 0.5), np.array([[4.05]]))
    assert_allclose(d_ridge_obj_th0(X, Y, th, th0, 100.0), np.array([[4.05]]))



Xt, Yt = downwards_line()

def test_stoc_grad_desc():
    ans = np.array([[-1.16237458], [0.48963195]])
    assert_allclose(stoc_grad_desc(Xt, Yt, J, dJ, np.array([[0.],[0.]]), 0.001, 10000), ans)
    ans = np.array([[-0.8093192], [0.34010595]])
    assert_allclose(stoc_grad_desc(Xt, Yt, J, dJ, np.array([[0.],[0.]]), 0.005, 1000), ans)
