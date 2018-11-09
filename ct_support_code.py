# You are expected to use this support code if you are writing code in Python.
# You may want to write:
# from ct_support_code import *
# at the top of your answers

# You will need numpy and scipy:
import numpy as np
from scipy.optimize import minimize

def params_unwrap(param_vec, shapes, sizes):
    """Helper routine for minimize_list"""
    args = []
    pos = 0
    for i in range(len(shapes)):
        sz = sizes[i]
        args.append(param_vec[pos:pos+sz].reshape(shapes[i]))
        pos += sz
    return args

def params_wrap(param_list):
    """Helper routine for minimize_list"""
    param_list = [np.array(x) for x in param_list]
    shapes = [x.shape for x in param_list]
    sizes = [x.size for x in param_list]
    param_vec = np.zeros(sum(sizes))
    pos = 0
    for param in param_list:
        sz = param.size
        param_vec[pos:pos+sz] = param.ravel()
        pos += sz
    unwrap = lambda pvec: params_unwrap(pvec, shapes, sizes)
    return param_vec, unwrap

def minimize_list(cost, init_list, args):
    """Optimize a list of arrays (wrapper of scipy.optimize.minimize)

    The input function "cost" should take a list of parameters,
    followed by any extra arguments:
        cost(init_list, *args)
    should return the cost of the initial condition, and a list in the same
    format as init_list giving gradients of the cost wrt the parameters.

    The options to the optimizer have been hard-coded. You may wish
    to change disp to True to get more diagnostics. You may want to
    decrease maxiter while debugging. Although please report all results
    in Q2-5 using maxiter=500.

    The Matlab code comes with a different optimizer, so won't give the same
    results.
    """
    opt = {'maxiter': 500, 'disp': False}
    init, unwrap = params_wrap(init_list)
    def wrap_cost(vec, *args):
        E, params_bar = cost(unwrap(vec), *args)
        vec_bar, _ = params_wrap(params_bar)
        return E, vec_bar
    res = minimize(wrap_cost, init, args, 'L-BFGS-B', jac=True, options=opt)
    return unwrap(res.x)


def linreg_cost(params, X, yy, alpha):
    """Regularized least squares cost function and gradients

    Can be optimized with minimize_list -- see fit_linreg_gradopt for a
    demonstration.

    Inputs:
    params: tuple (ww, bb): weights ww (D,), bias bb scalar
         X: N,D design matrix of input features
        yy: N,  real-valued targets
     alpha: regularization constant

    Outputs: (E, [ww_bar, bb_bar]), cost and gradients
    """
    # Unpack parameters from list
    ww, bb = params

    # forward computation of error
    ff = np.dot(X, ww) + bb
    res = ff - yy
    E = np.dot(res, res) + alpha*np.dot(ww, ww)

    # reverse computation of gradients
    ff_bar = 2*res
    bb_bar = np.sum(ff_bar)
    ww_bar = np.dot(X.T, ff_bar) + 2*alpha*ww

    return E, [ww_bar, bb_bar]

def fit_linreg_gradopt(X, yy, alpha):
    """
    fit a regularized linear regression model with gradient opt

         ww, bb = fit_linreg_gradopt(X, yy, alpha)

     Find weights and bias by using a gradient-based optimizer
     (minimize_list) to improve the regularized least squares cost:

       np.sum(((np.dot(X,ww) + bb) - yy)**2) + alpha*np.dot(ww,ww)

     Inputs:
             X N,D design matrix of input features
            yy N,  real-valued targets
         alpha     scalar regularization constant

     Outputs:
            ww D,  fitted weights
            bb     scalar fitted bias
    """
    D = X.shape[1]
    args = (X, yy, alpha)
    init = (np.zeros(D), np.array(0))
    ww, bb = minimize_list(linreg_cost, init, args)
    return ww, bb


def pca_zm_proj(X, K=None):
    """return PCA projection matrix for zero mean data

    Inputs:
        X N,D design matrix of input features -- must be zero mean
        K     how many columns to return in projection matrix

    Outputs:
        V D,K matrix to apply to X or other matrices shifted in same way.
    """
    if np.max(np.abs(np.mean(X,0))) > 1e-9:
        raise ValueError('Data is not zero mean.')
    if K is None:
        K = X.shape[1]
    E, V = np.linalg.eig(np.dot(X.T, X))
    idx = np.argsort(E)[::-1]
    V = V[:, idx[:K]] # D,K
    return V


def logreg_cost(params, X, yy, alpha):
    """Regularized logistic regression cost function and gradients

    Can be optimized with minimize_list -- see fit_linreg_gradopt for a
    demonstration of fitting a similar function.

    Inputs:
    params: tuple (ww, bb): weights ww (D,), bias bb scalar
         X: N,D design matrix of input features
        yy: N,  real-valued targets
     alpha: regularization constant

    Outputs: (E, [ww_bar, bb_bar]), cost and gradients
    """
    # Unpack parameters from list
    ww, bb = params

    # Force targets to be +/- 1
    yy = 2*(yy==1) - 1

    # forward computation of error
    aa = yy*(np.dot(X, ww) + bb)
    sigma = 1/(1 + np.exp(-aa))
    E = -np.sum(np.log(sigma)) + alpha*np.dot(ww, ww)

    # reverse computation of gradients
    aa_bar = sigma - 1
    bb_bar = np.dot(aa_bar, yy)
    ww_bar = np.dot(X.T, yy*aa_bar) + 2*alpha*ww

    return E, (ww_bar, bb_bar)


def fit_logreg(X, yy, alpha):
    D = X.shape[1]
    args = (X, yy, alpha)
    init = (np.zeros(D), np.array(0))
    ww, bb = minimize_list(logreg_cost, init, args)
    return ww, bb


def nn_cost(params, X, yy=None, alpha=None):
    """NN_COST simple neural network cost function and gradients, or predictions

           E, params_bar = nn_cost([ww, bb, V, bk], X, yy, alpha)
                    pred = nn_cost([ww, bb, V, bk], X)

     Cost function E can be minimized with minimize_list

     Inputs:
             params (ww, bb, V, bk), where:
                    --------------------------------
                        ww K,  hidden-output weights
                        bb     scalar output bias
                         V K,D hidden-input weights
                        bk K,  hidden biases
                    --------------------------------
                  X N,D input design matrix
                 yy N,  regression targets
              alpha     scalar regularization for weights

     Outputs:
                     E  sum of squares error
            params_bar  gradients wrt params, same format as params
     OR
               pred N,  predictions if only params and X are given as inputs
    """
    # Unpack parameters from list
    ww, bb, V, bk = params

    # Forwards computation of cost
    A = np.dot(X, V.T) + bk[None,:] # N,K
    P = 1 / (1 + np.exp(-A)) # N,K
    F = np.dot(P, ww) + bb # N,
    if yy is None:
        # user wants prediction rather than training signal:
        return F
    res = F - yy # N,
    E = np.dot(res, res) + alpha*(np.sum(V*V) + np.dot(ww,ww)) # 1x1

    # Reverse computation of gradients
    F_bar = 2*res # N,
    ww_bar = np.dot(P.T, F_bar) + 2*alpha*ww # K,
    bb_bar = np.sum(F_bar) # scalar
    P_bar = np.dot(F_bar[:,None], ww[None,:]) # N,
    A_bar = P_bar * P * (1 - P) # N,
    V_bar = np.dot(A_bar.T, X) + 2*alpha*V # K,
    bk_bar = np.sum(A_bar, 0)

    return E, (ww_bar, bb_bar, V_bar, bk_bar)


def fit_nn(params, X, yy=None, alpha=None):
    args = (X, yy, alpha)
    ww_bar, bb_bar, V_bar, bk_bar = minimize_list(nn_cost, params, args)

    return (ww_bar, bb_bar, V_bar, bk_bar)

'''
def gradient_descent_cost(params, X, yy=None, iter=100, learning_rate=0.0001):
    m_current, b_current = params
    N = float(len(yy))
    for i in range(iter):
        # print ("X shape:", X.shape)
        # print ("yy shape:", yy.shape)
        # print ("ww shape:", m_current.shape)
        # print ("bb shape:", b_current.shape, b_current)
        y_current = X.dot(m_current) + b_current
        E = sum([data**2 for data in (yy - y_current)]) / N
        m_gradient = -(2/float(N)) * sum(X.T.dot(yy - y_current))
        b_gradient = -(2/float(N)) * sum(yy - y_current)
        m_current = m_current - (learning_rate * m_gradient)
        b_current = b_current - (learning_rate * b_gradient)
    return m_current, b_current, E
'''

def gradient_descent(X, yy, ww, bb, learning_rate):
    bb_gradient = np.array(0.)
    ww_gradient = np.zeros(X.shape[1])
    N = X.shape[0]
    for i in range(0, N):
        x = X[i][:]
        y = yy[i]
        bb_gradient += -(2/float(N)) * (y - (x.dot(ww) + bb))
        ww_gradient += -(2/float(N)) * x * (y - (x.dot(ww) + bb))
    bb_bar = bb - (learning_rate * bb_gradient)
    ww_bar = ww - (learning_rate * ww_gradient)

    return [ww_bar, bb_bar]


def fit_gd(params, X, yy=None, iter=500, learning_rate=0.01):
    ww = params[0]
    bb = params[1]
    params_arr = []
    for i in range(iter):
        print (i)
        ww, bb = gradient_descent(X, yy, ww, bb, learning_rate)
        params_arr.append((ww, bb))

    return [ww, bb, params_arr]


# def grad_test(params, X, yy, learning_rate):
#     ww, bb = params
#     bb_gradient = np.array(0.)
#     ww_gradient = np.zeros(X.shape[1])
#     N = X.shape[0]
#
#     ff = np.dot(X, ww) + bb
#     res = ff - yy
#     E = np.dot(res, res)
#
#     for i in range(0, N):
#         x = X[i][:]
#         y = yy[i]
#         bb_gradient += -(2/float(N)) * (y - (x.dot(ww) + bb))
#         ww_gradient += -(2/float(N)) * x * (y - (x.dot(ww) + bb))
#     bb_bar = bb - (learning_rate * bb_gradient)
#     ww_bar = ww - (learning_rate * ww_gradient)
#
#     return E, [ww_bar, bb_bar]
#
#
# def fit_grad_test(params, X, yy, lr):
#     args = (X, yy, lr)
#     ww, bb = minimize_list(grad_test, params, args)
#     return ww, bb
# def linear_regression(X, y, m_current=0, b_current=0, epochs=1000, learning_rate=0.0001):
#      N = float(len(y))
#      for i in range(epochs):
#           y_current = (m_current * X) + b_current
#           cost = sum([data**2 for data in (y-y_current)]) / N
#           m_gradient = -(2/N) * sum(X * (y - y_current))
#           b_gradient = -(2/N) * sum(y - y_current)
#           m_current = m_current - (learning_rate * m_gradient)
#           b_current = b_current - (learning_rate * b_gradient)
#      return m_current, b_current, cost
