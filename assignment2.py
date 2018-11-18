import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from functools import reduce
from ct_support_code import fit_linreg_gradopt, pca_zm_proj, logreg_cost, fit_logreg, nn_cost, fit_nn, fit_gd
import scipy
import scipy.io
import math
import timeit

'''
EXERCISE 1
'''
def process_data():
    ct_data = scipy.io.loadmat('ct_data.mat', squeeze_me=True)

    X_train = ct_data['X_train']        # (40754, 384)
    X_val = ct_data['X_val']            # (5785, 384)
    X_test = ct_data['X_test']          # (6961, 384)
    y_train = ct_data['y_train']        # (40754,)
    y_val = ct_data['y_val']            # (5785,)
    y_test = ct_data['y_test']          # (6961,)

    # print ("X_train shape: ", X_train.shape)
    # print ("X_val shape: ", X_val.shape)
    # print ("X_test shape: ", X_test.shape)
    # print ("y_train shape: ", y_train.shape)
    # print ("y_val shape: ", y_val.shape)
    # print ("y_test shape: ", y_test.shape)

    # Ex 1.a
    # compute_standard_error(y_train=y_train, y_val=y_val)

    # Ex 1.b
    X_train, X_val, X_test = remove_unnecessary_features(X_train=X_train, X_val=X_val, X_test=X_test)   # (D = 373)

    # Ex 2
    # set_linear_regression_baseline(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

    # Ex 3
    # decrease_and_increase_input(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

    # Ex 4
    # invent_classification(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

    # Ex 5
    neural_network(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

    # Ex 6
    # gradient_descent(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, X_test=X_test, y_test=y_test)


'''
EXERCISE 1.a
'''
def compute_standard_error(y_train, y_val):
    mean_y_train = np.mean(y_train)                 # -9.13868774539957e-15
    print ("mean_y_train: ", mean_y_train)
    if round(mean_y_train, 10) == 0:
        print ("The mean is approximately 0!")
    else:
        print ("The mean is not approximately 0!")

    mean_y_val = np.mean(y_val)                     # -0.2160085093241599
    print ("mean_y_val: ", mean_y_val)
    if round(mean_y_val, 10) == 0:
        print ("The mean is approximately 0!")
    else:
        print ("The mean is not approximately 0!")

    std_y_train = np.std(y_train[:len(y_val)])
    print ("std_y_train: ", std_y_train)
    print ("std_y_train for entire y_train: ", np.std(y_train))
    std_y_val = np.std(y_val)
    print ("std_y_val: ", std_y_val)

    sem_y_train = std_y_train / np.sqrt(len(y_val))
    print ("sem_y_train: ", sem_y_train)
    sem_y_val = std_y_val / np.sqrt(len(y_val))
    print ("sem_y_val: ", sem_y_val)


'''
EXERCISE 1.b
'''
def remove_unnecessary_features(X_train, X_val, X_test):
    # Remove constant features
    X_train_delete_1 = remove_constant_features(X=X_train)
    X_val_delete_1 = remove_constant_features(X=X_val)
    X_test_delete_1 = remove_constant_features(X=X_test)

    # Remove duplicate features
    X_train_delete_2 = remove_duplicates_features(X=X_train)
    X_val_delete_2 = remove_duplicates_features(X=X_val)
    X_test_delete_2 = remove_duplicates_features(X=X_test)

    X_train_delete = X_train_delete_1 + X_train_delete_2
    X_val_delete = X_val_delete_1 + X_val_delete_2
    X_test_delete = X_test_delete_1 + X_test_delete_2

    to_delete = reduce(np.intersect1d, (X_train_delete, X_val_delete, X_test_delete))
    print (to_delete)
    print (len(to_delete))

    # Filter categories
    X_train_first, X_train_second = filter_remove_categories(to_delete, X_train_delete_1, X_train_delete_2)
    print ("X_train first: ", X_train_first)
    print ("X_train second: ", X_train_second)
    X_val_first, X_val_second = filter_remove_categories(to_delete, X_val_delete_1, X_val_delete_2)
    print ("X_val first: ", X_val_first)
    print ("X_val second: ", X_val_second)
    X_test_first, X_test_second = filter_remove_categories(to_delete, X_test_delete_1, X_test_delete_2)
    print ("X_test first: ", X_test_first)
    print ("X_test second: ", X_test_second)

    # Delete bad columns
    X_train = np.delete(X_train, to_delete, axis=1)         # (40754, 373)
    # print ("X_train shape after clean: ", X_train.shape)
    X_val = np.delete(X_val, to_delete, axis=1)             # (5785, 373)
    # print ("X_val shape after clean: ", X_val.shape)
    X_test = np.delete(X_test, to_delete, axis=1)           # (6961, 373)
    # print ("X_test shape after clean: ", X_test.shape)

    return X_train, X_val, X_test


def remove_constant_features(X):
    X_delete_1 = []
    for i in range(X.shape[1]):
        if (len(set(X[:,i])) == 1):
            X_delete_1.append(i)
    return X_delete_1


def remove_duplicates_features(X):
    unique, train_indices = np.unique(X, return_index=True, axis=1)
    X_delete_2 = list(set(range(X.shape[1])) - set(train_indices))
    return X_delete_2


def filter_remove_categories(to_delete, X_delete_1, X_delete_2):
    X_first = []
    X_second = []
    for i in to_delete:
        if i in X_delete_1:
            X_first.append(i)
        if i in X_delete_2:
            X_second.append(i)
    return X_first, X_second


'''
EXERCISE 2
'''
def set_linear_regression_baseline(X_train, X_val, y_train, y_val):
    # For training
    w_ls, b_ls = least_squares(X_train, y_train)
    err = compute_err(X=X_train, yy=y_train, ww=w_ls, bb=b_ls)
    print ("ERROR Least Squares:", err)                 # 0.35524169481074713
    X_prime, y_prime, w_prime = fit_linreg(X_train, y_train, 10)
    err = compute_err(X=X_prime, yy=y_prime, ww=w_prime)
    print ("ERROR Regression with Regularization:", err)# 0.3563466330636692
    w_grad, b_grad = fit_linreg_gradopt(X_train, y_train, 10)
    err = compute_err(X=X_train, yy=y_train, ww=w_grad, bb=b_grad)
    print ("ERROR Gradient-Based Optimizer:", err)      # 0.35575973762757745

    # For validation
    err = compute_err(X=X_val, yy=y_val, ww=w_ls, bb=b_ls)
    print ("ERROR Least Squares:", err)                 # 0.4182210942058271
    X_prime, y_prime, dummy = fit_linreg(X_val, y_val, 10)
    err = compute_err(X=X_prime, yy=y_prime, ww=w_prime)
    print ("ERROR Regression with Regularization:", err)# 0.4202908632281942
    err = compute_err(X=X_val, yy=y_val, ww=w_grad, bb=b_grad)
    print ("ERROR Gradient-Based Optimizer:", err)      # 0.42060375407081924


def fit_linreg(X, yy, alpha):
    N = X.shape[0]
    D = X.shape[1]

    yy_prime = np.concatenate([yy, np.zeros(D)])
    alphaI = np.sqrt(alpha) * np.identity(D)
    X_part = np.concatenate([X, alphaI], axis=0)
    X_bias = np.ones(N+D).reshape(N+D, 1)
    X_bias[-D:] = 0
    X_prime = np.concatenate([X_bias, X_part], axis = 1)
    w_prime = np.linalg.lstsq(X_prime, yy_prime, rcond=0)[0]

    return X_prime, yy_prime, w_prime


def least_squares(X, yy):
    X_bias = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)
    w_bias = np.linalg.lstsq(X_bias, yy, rcond=0)[0];

    return w_bias[1:], w_bias[0]


def compute_err(X, yy, ww, bb=0):
    y_predicted = X.dot(ww) + bb
    return root_mean_square_error(y_expected=yy, y_predicted=y_predicted)


def root_mean_square_error(y_expected, y_predicted):
    sum = 0
    for i in range(len(y_expected)):
        sum += ((y_expected[i] - y_predicted[i]) ** 2)

    return np.sqrt(sum/len(y_expected))


'''
EXERCISE 3
'''
def decrease_and_increase_input(X_train, X_val, y_train, y_val):
    # Ex 3.a
    '''
    # For training
    V = pca(X=X_train, yy=y_train, alpha=10, K=10)
    X_reduced = X_train.dot(V)
    X_prime, yy_prime, w_prime = fit_linreg(X_reduced, y_train, 10)
    err = compute_err(X=X_prime, yy=yy_prime, ww=w_prime)
    print ("Training error for K = 10: ", err)  # 0.5729503891901699
    X_reduced = X_val.dot(V)
    X_prime, yy_prime, dummy = fit_linreg(X_reduced, y_val, 10)
    err = compute_err(X=X_prime, yy=yy_prime, ww=w_prime)
    print ("Validation error for K = 10: ", err)# 5719793306418868

    # For validation
    V = pca(X=X_train, yy=y_train, alpha=10, K=100)
    X_reduced = X_train.dot(V)
    X_prime, yy_prime, w_prime = fit_linreg(X_reduced, y_train, 10)
    err = compute_err(X=X_prime, yy=yy_prime, ww=w_prime)
    print ("Training error for K = 100: ", err) # 0.410527496797554
    X_reduced = X_val.dot(V)
    X_prime, yy_prime, dummy = fit_linreg(X_reduced, y_val, 10)
    err = compute_err(X=X_prime, yy=yy_prime, ww=w_prime)
    print ("Validation error for K = 100: ", err)# 0.4317438278524941
    '''

    # Ex 3.b
    '''
    X_prime, yy_prime, w_prime = histogram(X=X_train, yy=y_train, alpha=10)
    err = compute_err(X=X_prime, yy=yy_prime, ww=w_prime)
    print ("Training error: ", err)             # 0.3157491192020929
    X_prime, yy_prime, dummy = histogram(X=X_val, yy=y_val, alpha=10)
    err = compute_err(X=X_prime, yy=yy_prime, ww=w_prime)
    print ("Validation error: ", err)           # 0.3570051802759181
    '''


'''
EXERCISE 3.a
'''
def pca(X, yy, alpha, K):
    X_mu = np.mean(X, 0)
    X_centred = X - X_mu

    return pca_zm_proj(X=X_centred, K=K)


'''
EXERCISE 3.b
'''
def histogram(X, yy, alpha):
    # 46th feature - histogram
    # plt.clf()
    # plt.hist(X[:,45], bins= 20) #len(list(set(X[:,46])))//100)
    # plt.show()
    count_0, count_25, count_0_per, count_25_per = compute_percentage(X[:,46])
    print('Count of O ({0}); Count of -0.25 ({1}) for 46th feature'.format(count_0, count_25))
    print('Percentage of O ({0}); Percentage of -0.25 ({1}) for 46th feature'.format(count_0_per, count_25_per))

    # all training data - histogram
    # plt.clf()
    # plt.hist(np.ravel(X), bins= 20)
    # plt.show()
    count_0, count_25, count_0_per, count_25_per = compute_percentage(np.ravel(X))
    print('Count of O ({0}); Count of -0.25 ({1}) for all data'.format(count_0, count_25))
    print('Percentage of O ({0}); Percentage of -0.25 ({1}) for all data'.format(count_0_per, count_25_per))


    aug_fn = lambda X: np.concatenate([X, X==0, X<0], axis=1)
    X_prime = aug_fn(X)

    # all training data - histogram (after )
    # plt.clf()
    # plt.hist(np.ravel(X_prime), bins= 20)
    # plt.show()
    count_0, count_25, count_0_per, count_25_per = compute_percentage(np.ravel(X_prime))
    print('Count of O ({0}); Count of -0.25 ({1}) for all data'.format(count_0, count_25))
    print('Percentage of O ({0}); Percentage of -0.25 ({1}) for all data'.format(count_0_per, count_25_per))

    return fit_linreg(X_prime, yy, alpha)


def compute_percentage(X):
    count_0 = 0
    count_25 = 0
    for i in X:
        if i == 0:
            count_0 += 1
        elif i == -0.25:
            count_25 += 1

    count_0_per = count_0 * 100 / len(X)
    count_25_per = count_25 * 100 / len(X)
    return count_0, count_25, count_0_per, count_25_per


'''
EXERCISE 4
'''
def invent_classification(X_train, X_val, y_train, y_val):
    K = 10          # number of thresholded classification problems to fit
    # For training
    ww_log, bb_log = use_logreg(X=X_train, yy=y_train, alpha=10, K=10)
    X_reduced = X_train.dot(ww_log) + bb_log
    X_sigmoid = 1 / (1 + np.exp(-X_reduced))
    X_prime, yy_prime, w_prime = fit_linreg(X=X_sigmoid, yy=y_train, alpha=10)
    err = compute_err(X=X_prime, yy=yy_prime, ww = w_prime)
    print ("ERROR Logistic Regression: ", err)  # 0.13953545921776178

    # For validation
    X_reduced = X_val.dot(ww_log) + bb_log
    X_sigmoid = 1 / (1 + np.exp(-X_reduced))
    X_prime, yy_prime, dummy = fit_linreg(X=X_sigmoid, yy=y_val, alpha=10)
    err = compute_err(X=X_prime, yy=yy_prime, ww = w_prime)
    print ("ERROR Logistic Regression: ", err)  # 0.25687979969894703


def use_logreg(X, yy, alpha, K):
    mx = np.max(yy)
    mn = np.min(yy)
    hh = (mx-mn)/(K+1)
    thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)
    ww_array = []
    bb_array = []
    for kk in range(K):
        labels = yy > thresholds[kk]
        ww, bb = fit_logreg(X, labels, alpha)
        ww_array.append(ww)
        bb_array.append(bb)

    ww = np.column_stack(ww_array)
    bb = np.column_stack(bb_array)

    return ww, bb


'''
EXERCISE 5
'''
def neural_network(X_train, X_val, y_train, y_val):
    # Random Initialization
    params = random_initialization(X_train.shape[1], 10)
    new_params = fit_nn(params=params, X=X_train, yy=y_train, alpha=10)
    y_predicted = nn_cost(params=new_params, X=X_train, alpha=10)
    err = root_mean_square_error(y_expected=y_train, y_predicted=y_predicted)
    print ("ERR Training Random:", err)            # 0.10272908967311808
    y_predicted = nn_cost(params=new_params, X=X_val, alpha=10)
    err = root_mean_square_error(y_expected=y_val, y_predicted=y_predicted)
    print ("ERR Validation Random:", err)          # 0.26861494023916044

    # Q4 Initialization
    params = q4_initialization(X_train, y_train, 10, 10)
    new_params = fit_nn(params=params, X=X_train, yy=y_train, alpha=10)
    y_predicted = nn_cost(params=new_params, X=X_train, alpha=10)
    err = root_mean_square_error(y_expected=y_train, y_predicted=y_predicted)
    print ("ERR Training Ex4:", err)                # 0.10446102065201333
    y_predicted = nn_cost(params=new_params, X=X_val, alpha=10)
    err = root_mean_square_error(y_expected=y_val, y_predicted=y_predicted)
    print ("ERR Validation Ex4:", err)              # 0.2574382823541626


def q4_initialization(X, yy, alpha, K):
    V, bk = use_logreg(X=X, yy=yy, alpha=alpha, K=K)
    X_reduced = X.dot(V) + bk
    X_sigmoid = 1 / (1 + np.exp(-X_reduced))
    X_prime, yy_prime, w_prime = fit_linreg(X=X_sigmoid, yy=yy, alpha=alpha)
    ww = w_prime[1:]
    bb = w_prime[0]

    return (ww, bb, V.T, bk.T.reshape(K,))

def random_initialization(D, K):
    V = np.random.randn(K, D)
    bk = np.random.randn(K)
    ww = np.random.randn(K)
    bb = np.random.randn(1)[0]

    return (ww, bb, V, bk)


'''
EXERCISE 6
'''
def gradient_descent(X_train, X_val, y_train, y_val, X_test, y_test):
    init = (np.zeros(X_train.shape[1]), np.array(0))
    ww, bb, params_arr = fit_gd(init, X=X_train, yy=y_train, iter=500, learning_rate=0.05)

    errors = plot_errors(params_arr = params_arr, X=X_train, yy=y_train)
    plot_output(X=X_train, yy=y_train, ww=ww, bb=bb)
    print ("ERROR GD Training:", errors[-1])        # 0.3704747114536849

    errors = plot_errors(params_arr = params_arr, X=X_val, yy=y_val)
    plot_output(X=X_val, yy=y_val, ww=ww, bb=bb)
    print ("ERROR GD Validation:", errors[-1])      # 0.43155709194018754

    errors = plot_errors(params_arr = params_arr, X=X_test, yy=y_test)
    plot_output(X=X_test, yy=y_test, ww=ww, bb=bb)
    print ("ERROR GD Test:", errors[-1])            # 0.4297184947308366

    # lr = 0.01
    # ERROR GD Training: 0.4034079522761693
    # ERROR GD Validation: 0.4410599893520035

    # lr = 0.02
    # ERROR GD Training: 0.38609127137182075
    # ERROR GD Validation: 0.43251353468775006

    # lr = 0.05
    # ERROR GD Training: 0.3704747114536849
    # ERROR GD Validation: 0.43155709194018754
    # ERROR GD Test: 0.4297184947308366

def plot_errors(params_arr, X, yy):
    errors = []
    for ww, bb in params_arr:
        y_predicted = X.dot(ww) + bb
        err = root_mean_square_error(y_expected=yy, y_predicted=y_predicted)
        errors.append(err)

    grid = np.linspace(1, len(errors), len(errors))

    plt.clf()
    plt.plot(grid, errors, 'r-')
    plt.legend()
    plt.ylabel("error")
    plt.xlabel("#iterations")
    plt.show()

    return errors

def plot_output(X, yy, ww, bb=0):
    y_predicted = X.dot(ww) + bb
    grid = np.linspace(1, len(yy), len(yy))

    plt.clf()
    plt.plot(grid, yy, 'b-', label='Real')
    plt.plot(grid, y_predicted, 'r-', label='Fitting', alpha=0.9)
    plt.legend()
    plt.ylabel("output")
    plt.xlabel("#points")
    plt.show()

    # return root_mean_square_error(y_expected=yy, y_predicted=y_predicted)




if __name__ == '__main__':
    process_data()
