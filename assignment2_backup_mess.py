import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from functools import reduce
from ct_support_code import fit_linreg_gradopt, pca_zm_proj, logreg_cost, fit_logreg, nn_cost
import scipy
import scipy.io
import math

'''
EXERCISE 1
'''
def process_data():
    ct_data = scipy.io.loadmat('ct_data.mat', squeeze_me=True)
    # print (ct_data)
    # pprint(ct_data)         # print the elements from dictionary

    X_train = ct_data['X_train']        # (40754, 384)
    X_val = ct_data['X_val']            # (5785, 384)
    X_test = ct_data['X_test']          # (6961, 384)
    y_train = ct_data['y_train']        # (40754,)
    y_val = ct_data['y_val']            # (5785,)
    y_test = ct_data['y_test']          # (6961,)

    print ("X_train shape: ", X_train.shape)
    print ("X_val shape: ", X_val.shape)
    print ("X_test shape: ", X_test.shape)
    print ("y_train shape: ", y_train.shape)
    print ("y_val shape: ", y_val.shape)
    print ("y_test shape: ", y_test.shape)

    # Ex 1.a
    # compute_standard_error(y_train=y_train, y_val=y_val)

    # Ex 1.b
    X_train, X_val, X_test = remove_unnecessary_features(X_train=X_train, X_val=X_val, X_test=X_test)   # (D = 373)

    # Ex 2
    # set_linear_regression_baseline(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

    # Ex 3
    # decrease_and_increase_input(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

    # Ex 4
    invent_classification(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

    # Ex 5
    # neural_network(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

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
    ### EXPLAIN THE THING WITH 95% for 2*sem
    ### EXPLAIN WHY THE BARS ARE MISLEADING HERE
    '''

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
    err_lstsq = least_squares(X_train, y_train)         # 0.35524169481074713
    print ("ERROR Least Squares: ", err_lstsq)
    err = fit_linreg(X_train, y_train, 10)              # 0.36915529643475953
    print ("ERROR Regression: ", err)
    ww, bb = fit_linreg_gradopt(X_train, y_train, 10)
    err_gbo = error_gradopt(X_train, y_train, ww, bb)   # 0.35575973762757745
    print ("ERROR Gradient-Based Optimizer: ", err_gbo)

    # For validation
    err_lstsq = least_squares(X_val, y_val)             # 0.18749417046818448
    print ("ERROR Least Squares: ", err_lstsq)
    err = fit_linreg(X_val, y_val, 10)                  # 0.29502328132809175
    print ("ERROR Regression: ", err)
    ww, bb = fit_linreg_gradopt(X_val, y_val, 10)
    err_gbo = error_gradopt(X_val, y_val, ww, bb)       # 0.1988673366957355
    print ("ERROR Gradient-Based Optimizer: ", err_gbo)

    '''
    The results obtained from the Gradient-Based Optimizer function for root
    mean square error were better that the ones obtained using regularization.
    More than that, the results from the Gradient-Based Optimizer were similar
    with the ones obtained using least squares without regularization.

    ### WHY IS THAT
    '''

def fit_linreg(X, yy, alpha):
    D = X.shape[1]

    yy_prime = np.concatenate([yy, np.zeros(D)])
    alphaI = alpha * np.identity(D)
    X_prime = np.concatenate([X, alphaI], axis=0)

    w_prime = np.linalg.lstsq(X_prime, yy_prime, rcond=0)[0]
    yy_prime_predicted = X_prime.dot(w_prime)

    return root_mean_square_error(y_expected=yy_prime, y_predicted=yy_prime_predicted)


def least_squares(X, yy):
    X_bias = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)
    w_bias = np.linalg.lstsq(X_bias, yy, rcond=0)[0];

    y_predicted = X_bias.dot(w_bias)

    return root_mean_square_error(y_expected=yy, y_predicted=y_predicted)

def error_gradopt(X, yy, ww, bb):
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

    err = pca(X=X_train, yy=y_train, alpha=10, K=10)
    print ("Training error for K = 10: ", err)
    err = pca(X=X_train, yy=y_train, alpha=10, K=100)
    print ("Training error for K = 100: ", err)
    err = pca(X=X_val, yy=y_val, alpha=10, K=10)
    print ("Validation error for K = 10: ", err)
    err = pca(X=X_val, yy=y_val, alpha=10, K=100)
    print ("Validation error for K = 100: ", err)


    # Ex 3.b
    '''
    err = histogram(X=X_train, yy=y_train, alpha=10)
    print ("Training error: ", err)
    err = histogram(X=X_val, yy=y_val, alpha=10)
    print ("Validation error: ", err)
    '''

'''
EXERCISE 3.a
'''
def pca(X, yy, alpha, K):
    X_mu = np.mean(X, 0)
    X_centred = X - X_mu

    V = pca_zm_proj(X=X_centred, K=K)
    X_reduced = X.dot(V)
    return fit_linreg(X_reduced, yy, alpha)

    '''
    ### EXPLAIN WHY ERROR IS WORSE WITH PCA (WITH DECREASING K)
    '''

    # Training error for K = 10:  0.5756376489484005
    # Training error for K = 100:  0.4153790239683063
    # Validation error for K = 10:  (0.5673115345895073+0j)
    # Validation error for K = 100:  (0.3363259856687762+0j)

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

    '''
    ### EXPLAIN WHY ERROR IS LOWER WHEN AUGMENTING THESE DATA
    '''

    # Training error:  0.3260341808564488
    # Validation error:  0.20646543050845095


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
    err = err_logreg(X_train, y_train, 10, K)        # 0.4223330824826547
    print ("ERROR Logistic Regression: ", err)

    # For validation
    err = err_logreg(X_val, y_val, 10, K)            # 0.36335171565304947
    print ("ERROR Logistic Regression: ", err)

    '''
    ### DIFFERENCES BETWEEN LOGISTIC REGRESSION AND PCA AS PERFORMANCE
    '''


def err_logreg(X, yy, alpha, K):
    ww, bb = use_logreg(X, yy, alpha, K)
    X_reduced = X.dot(ww) + bb

    return fit_linreg(X_reduced, yy, alpha)

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
    q4_initialization(X_train, y_train, 10, 10)


def q4_initialization(X, yy, alpha, K):
    V, bk = use_logreg(X, yy, alpha, K)
    X_reduced = X.dot(V) + bk

    ww, bb = fit_linreg_gradopt(X_reduced, yy, alpha)

    # X_bias = np.concatenate([np.ones((X_reduced.shape[0],1)), X_reduced], axis=1)
    # w_bias = np.linalg.lstsq(X_bias, yy, rcond=0)[0];
    #
    # print ("w_bias shape:", w_bias.shape)
    #
    # ww = w_bias[1:]
    # bb = w_bias[0]

    # 47207.983806580836
    # 47207.07096537262 (alpha=10)

    print ("X shape:", X.shape)
    print ("yy shape:", yy.shape)
    print ("ww shape:", ww.shape)
    print ("bb shape:", bb.shape, bb)
    print ("V shape:", V.T.shape)
    print ("bk shape:", bk.T.shape)

    params = (ww, bb, V.T, bk.T.reshape(K,))

    err, params_bar = nn_cost(params=params, X=X, yy=yy, alpha=alpha)
    print ("NN ERR: ", err)
    ww_bar, bb_bar, V_bar, bk_bar = params_bar
    print(ww_bar.shape, bb_bar.shape, V_bar.shape, bk_bar.shape)

    # y_predicted = (X.dot(V_bar.T) + bk_bar.T).dot(ww_bar) + bb_bar
    #
    # print (root_mean_square_error(y_expected=yy, y_predicted=y_predicted))


def get_parameters_linreg(X, yy, alpha):
    D = X.shape[1]

    yy_prime = np.concatenate([yy, np.zeros(D)])
    alphaI = alpha * np.identity(D)
    X_prime = np.concatenate([X, alphaI], axis=0)

    w_prime = np.linalg.lstsq(X_prime, yy_prime, rcond=0)[0]
    # yy_prime_predicted = X_prime.dot(w_prime)

    print ("SHAPE: ", w_prime.shape)





if __name__ == '__main__':
    process_data()
