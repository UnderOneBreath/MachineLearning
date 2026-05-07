import numpy as np
from numpy.testing import assert_allclose

from lab03 import rv, cv, f1, df1, f2, df2
from lab03 import super_simple_separable, separable_medium
from lab03 import grad_desc, num_grad, num_grad_desc
from lab03 import hinge, hinge_loss, svm_obj
from lab03 import d_hinge, d_hinge_loss_th, d_hinge_loss_th0, d_svm_obj_th, d_svm_obj_th0, svm_obj_grad, svm_grad_desc


def test_grad_desc():
    ans = grad_desc(f1, df1, cv([0.]), 0.1, 1000)
    assert_allclose(ans, cv([-1.5]))
    ans = grad_desc(f2, df2, cv([0., 0.]), 0.01, 1000)
    assert_allclose(ans, cv([-2.2058239, 3.20582389]))


def test_num_grad():
    ans = num_grad(f1)(cv([0.]))
    assert_allclose(ans, cv([12.]))

    ans = num_grad(f1)(cv([0.1]))
    assert_allclose(ans, cv([12.8]))

    ans = num_grad(f2)(cv([0., 0.]))
    assert_allclose(ans, cv([6.999999, -2.]))

    ans = num_grad(f2)(cv([0.1, -0.1]))
    assert_allclose(ans, cv([4.7739994, -2.]))


def test_num_grad_desc():
    ans = num_grad_desc(f1, cv([0.]), 0.1, 1000)
    assert_allclose(ans, cv([-1.5]))
    ans = num_grad_desc(f2, cv([0., 0.]), 0.01, 1000)
    assert_allclose(ans, cv([-2.20582371, 3.20582369]))


sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])


def test_hinge():
    assert_allclose(hinge(rv([0.])), 1.)
    assert_allclose(hinge(rv([0., 0.3, 0.7, 1., 1.2])), rv([1., 0.7, 0.3, 0., 0.]))

def test_hinge_loss():
    X, Y = super_simple_separable()
    th, th0 = sep_e_separator
    assert_allclose(hinge_loss(X, Y, th, th0), rv([0., 0., 0., 0.]))
    th, th0 = np.array([[-0.3], [1.2]]), np.array([[-2.1]])
    assert_allclose(hinge_loss(X, Y, th, th0), rv([0., 0.4, 0., 1.3]))

def test_svm_obj():
    X, Y = super_simple_separable()
    th, th0 = sep_e_separator
    assert_allclose(svm_obj(X, Y, th, th0, 0.1), 0.15668396890496103)
    assert_allclose(svm_obj(X, Y, th, th0, 0.0), 0.0)

    th, th0 = np.array([[-0.3], [1.2]]), np.array([[-2.1]])
    assert_allclose(svm_obj(X, Y, th, th0, 0.1), 0.5780000000000001)
    assert_allclose(svm_obj(X, Y, th, th0, 0.0), 0.42500000000000004)

def test_d_hinge():
    assert_allclose(d_hinge(rv([71.])), 0.)
    assert_allclose(d_hinge(rv([-23.])), -1.)
    assert_allclose(d_hinge(rv([71., -23.])), rv([0., -1.]))

X1 = np.array([[1, 2, 3, 9, 10]])
Y1 = np.array([[1, 1, 1, -1, -1]])
th1, th10 = np.array([[-0.31202807]]), np.array([[1.834]])
X2 = np.array([[2, 3, 9, 12],
               [5, 2, 6, 5]])
Y2 = np.array([[1, -1, 1, -1]])
th2, th20 = np.array([[-3., 15.]]).T, np.array([[2.]])

def test_d_hinge_loss_th():
    assert_allclose(d_hinge_loss_th(X1, Y1, th1, th10), rv([0., 0., -3., 9., 0.]))
    assert_allclose(d_hinge_loss_th(X2[:,0:1], Y2[:,0:1], th2, th20), cv([0., 0.]))
    assert_allclose(d_hinge_loss_th(X2, Y2, th2, th20), np.array([[0., 3., 0., 12.], [0., 2., 0., 5.]]))

def test_d_hinge_loss_th0():
    assert_allclose(d_hinge_loss_th0(X1, Y1, th1, th10), rv([0., 0., -1., 1., 0.]))
    assert_allclose(d_hinge_loss_th0(X2[:,0:1], Y2[:,0:1], th2, th20), 0.)
    assert_allclose(d_hinge_loss_th0(X2, Y2, th2, th20), rv([0., 1., 0., 1.]))

def test_d_svm_obj_th():
    assert_allclose(d_svm_obj_th(X1, Y1, th1, th10, 0.01), 1.19375944)
    assert_allclose(d_svm_obj_th(X2[:,0:1], Y2[:,0:1], th2, th20, 0.01), cv([-0.06, 0.3]))
    assert_allclose(d_svm_obj_th(X2, Y2, th2, th20, 0.01), cv([3.69, 2.05]))

def test_d_svm_obj_th0():
    assert_allclose(d_svm_obj_th0(X1, Y1, th1, th10, 0.01), 0.)
    assert_allclose(d_svm_obj_th0(X2[:,0:1], Y2[:,0:1], th2, th20, 0.01), 0.)
    assert_allclose(d_svm_obj_th0(X2, Y2, th2, th20, 0.01), 0.5)

def test_grad_svm_obj():
    assert_allclose(svm_obj_grad(X1, Y1, th1, th10, 0.01), cv([1.19375944, 0.]))
    assert_allclose(svm_obj_grad(X2, Y2, th2, th20, 0.01), cv([3.69, 2.05, 0.5]))
    assert_allclose(svm_obj_grad(X2[:,0:1], Y2[:,0:1], th2, th20, 0.01), cv([-0.06, 0.3, 0.]))
    assert_allclose(svm_obj_grad(X1, Y1, th1, th10, 0.15), cv([1.10639158, 0.]))
    assert_allclose(svm_obj_grad(X2, Y2, th2, th20, 0.15), cv([2.85, 6.25, 0.5]))
    assert_allclose(svm_obj_grad(X2[:,0:1], Y2[:,0:1], th2, th20, 0.15), cv([-0.9, 4.5, 0.]))


def test_svm_grad_desc():
    X1, Y1 = super_simple_separable()
    X2, Y2 = separable_medium()
    assert_allclose(svm_grad_desc(X1, Y1, 0.0001, 0.0001, 1000), cv([-0.099899, 0.099899, 0.]))
    assert_allclose(svm_grad_desc(X1, Y1, 0.01, 0.0001, 1000), cv([-0.09980037, 0.09980037, 0.]))
    assert_allclose(svm_grad_desc(X2, Y2, 0.0001, 0.0001, 1000), cv([0.07492425, -0.02497475, 0.]))
    assert_allclose(svm_grad_desc(X2, Y2, 0.01, 0.0001, 1000), cv([0.07485027, -0.02495009, 0.]))

