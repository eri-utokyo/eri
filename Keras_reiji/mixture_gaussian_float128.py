from __future__ import division
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import argparse

eps = np.float128(1e-6)

def batch_dot(a, b):
    return np.sum(a*b, axis=len(a.shape)-1)

def gaussian(x, mu, sigma):
    A   = 1./((2*np.pi*(sigma**2))**(args.dimension/2))
    phi = np.exp(-batch_dot(x[:, np.newaxis, :]-mu, x[:, np.newaxis, :]-mu)/(2*sigma**2))
    return A*phi

def loglikelihood(x, w, mu, sigma):
    return np.sum(np.log(np.sum(w*gaussian(x, mu, sigma)+eps, axis=1)), axis=0)

def update(x, w, mu, sigma):
    _eta      = w*gaussian(x, mu, sigma)+eps
    eta       = _eta/np.sum(_eta, axis=1, keepdims=True)
    new_w     = np.mean(eta, axis=0)
    new_mu    = np.sum(eta[:, :, np.newaxis]*x[:, np.newaxis, :], axis=0)/np.sum(eta, axis=0, keepdims=True).T
    new_sigma = np.sqrt(np.sum(eta*batch_dot(x[:, np.newaxis, :]-mu, x[:, np.newaxis, :]-mu), axis=0)/(args.dimension*np.sum(eta, axis=0)))
    return new_w, new_mu, new_sigma

def w_init():
    return np.ones(shape=(args.mixture, ), dtype=np.float128)/args.mixture
    #return np.arange(1, args.mixture+1, dtype=np.float128)/np.sum(np.arange(1, args.mixture+1, dtype=np.float128))

def mean_concat(train, mixture):
    if mixture == 1:
        return np.mean(train, axis=0, keepdims=True)*np.random.normal(1, 1e-3)
    mu_init = np.concatenate((np.mean(train, axis=0, keepdims=True)*np.random.normal(1, 1e-3), np.mean(train, axis=0, keepdims=True)*np.random.normal(1, 1e-3)))
    for _ in range(1, mixture-1):
        mu_init = np.concatenate((mu_init, np.mean(train, axis=0, keepdims=True)*np.random.normal(1, 1e-3)))
    return mu_init

def sigma_init():
    return np.arange(1, args.mixture+1, dtype=np.float128)/np.sum(np.arange(1, args.mixture+1, dtype=np.float128))
    #return np.ones(shape=(args.mixture), dtype=np.float128)

def pred(x, w, mu, sigma):
    return np.sum(w*gaussian(x, mu, sigma), axis=1)

if __name__ == '__main__':
    perser = argparse.ArgumentParser()
    perser.add_argument('-d', '--dimension', default=784, type=int)
    perser.add_argument('-m', '--mixture', default=3, type=int)
    args = perser.parse_args()

    mnist                            = fetch_mldata('MNIST original')
    train_X                          = mnist.data.astype(np.float128)/255.
    train_y                          = mnist.target.astype(np.int32)
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2)
    train_X                          = [train_X[train_y == i] for i in range(10)]

    ws     = []
    mus    = []
    sigmas = []

    for i, train in enumerate(train_X):
        print "target: {}".format(i)
        w     = w_init()
        mu    = mean_concat(train, args.mixture)
        sigma = sigma_init()

        p_loglikelihood = -np.inf
        while loglikelihood(train, w, mu, sigma)-p_loglikelihood > loglikelihood(train, w, mu, sigma)*0.000000001 or p_loglikelihood < 0:
            p_loglikelihood = loglikelihood(train, w, mu, sigma)
            print "loglikelihood: {}".format(p_loglikelihood)
            w, mu, sigma = update(train, w, mu, sigma)
            print "sigma: {}".format(sigma)
        print "last: {}".format(loglikelihood(train, w, mu, sigma))


        ws.append(w)
        mus.append(mu)
        sigmas.append(sigma)

    predicts = []

    for i, (w, mu, sigma) in enumerate(zip(ws, mus, sigmas)):
        predicts.append(pred(test_X, w, mu, sigma))
    predicts = np.array(predicts)
    pred_y = predicts.argmax(axis=0)
    print "f1_score: {}".format(f1_score(test_y, pred_y, average='macro'))
    print "accuracy: {}".format(accuracy_score(test_y, pred_y))

