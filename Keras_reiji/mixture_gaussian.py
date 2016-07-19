from __future__ import division
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('mixture', 3, "Number of mixture gaussian")
flags.DEFINE_integer('dimension', 784, "Number of Dimension")

def gaussian(x, mu, sigma):
    """
    tf.expand_dims(x, 1): (len(train), 1, dimension)
    tf.expand_dims(x, 1)-mu: (len(train), mixture, dimension)
    x = tf.expand_dims(tf.expand_dims(x, 1)-mu, 2): (len(train), mixture, 1, dimension)
    y = tf.expand_dims(tf.expand_dims(x, 1)-mu, 3): (len(train), mixture, dimension, 1)
    input = tf.batch_matmul(x, y): (len(train), mixture, 1, 1)
    tf.reduce_sum(input, reduction_indices=[2, 3]): (len(train), mixture)
    """
    A = 1./((2*np.pi*sigma**2)**(FLAGS.dimension/2))
    phi = tf.exp(-tf.reduce_sum(tf.batch_matmul(tf.expand_dims(tf.expand_dims(x, 1)-mu, 2),
                                                tf.expand_dims(tf.expand_dims(x, 1)-mu, 3)
                                                ), reduction_indices=[2, 3]
                                )
                /(2*sigma**2)
                )
    return A*phi

def inference(x, mu_init):
    """
    x: (len(train), dimension)
    w: (mixture)
    mu: (mixture, dimension)
    sigma: (mixture)
    eta: (len(train), mixture)
    """
    w     = tf.Variable(tf.ones(shape=(FLAGS.mixture,), dtype=tf.float64)/FLAGS.mixture)
    mu    = tf.Variable(mu_init)
    sigma = tf.Variable(tf.ones(shape=(FLAGS.mixture,), dtype=tf.float64)*0.5)
    eps   = tf.constant(0.000001, dtype=tf.float64)
    _eta  = w*gaussian(x, mu, sigma)+eps
    eta   = _eta/tf.expand_dims(tf.reduce_sum(_eta, reduction_indices=1), 1)
    p     = tf.reduce_sum(_eta, reduction_indices=1)
    return w, mu, sigma, eta, p
def training(w, mu, sigma, eta):
    new_w     = tf.reduce_mean(eta, reduction_indices=0)
    new_mu    = tf.reduce_sum(tf.expand_dims(eta, 2)*tf.expand_dims(x, 1), reduction_indices=0)/tf.expand_dims(tf.reduce_sum(eta, reduction_indices=0), 1)
    new_sigma = tf.sqrt(tf.reduce_sum(eta*tf.reduce_sum(tf.batch_matmul(tf.expand_dims(tf.expand_dims(x, 1)-mu, 2),
                                                                        tf.expand_dims(tf.expand_dims(x, 1)-mu, 3)
                                                                        ), reduction_indices=[2, 3]
                                                        ), reduction_indices=0
                                      )/(FLAGS.dimension*tf.reduce_sum(eta, reduction_indices=0))
                        )
    
    update_w     = tf.assign(w, new_w)
    update_mu    = tf.assign(mu, new_mu)
    update_sigma = tf.assign(sigma, new_sigma)
    update = tf.group(update_w, update_mu, update_sigma)
    return update
def loglikelihood(x, mu, sigma, w):
    eps   = tf.constant(0.000000001, dtype=tf.float64)
    return tf.reduce_sum(tf.log(tf.reduce_sum(w*gaussian(x, mu, sigma)+eps, reduction_indices=1)), reduction_indices=0)

def mean_concat(train, mixture):
    np.random.normal(1, 0.001)
    mu_init = np.concatenate((np.mean(train, axis=0, keepdims=True)*np.random.normal(1, 0.01), np.mean(train, axis=0, keepdims=True)*np.random.normal(1, 0.01)))
    for _ in range(1, mixture-1):
        mu_init = np.concatenate((mu_init, np.mean(train, axis=0, keepdims=True)*np.random.normal(1, 0.01)))
    return mu_init


if __name__ == '__main__':
    mnist   = fetch_mldata('MNIST original')
    train_X = mnist.data.astype(np.float64)/255.
    train_y = mnist.target.astype(np.int32)
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2)
    train_X = [train_X[train_y == i] for i in range(10)]
    xs = []
    props = []
    updates = []
    loglikelihoods = []
    for i, train in enumerate(train_X):
        mu_init                 = mean_concat(train, FLAGS.mixture)
        x                       = tf.placeholder(dtype=tf.float64, shape=(None, FLAGS.dimension))
        w, mu, sigma, eta, prop = inference(x, mu_init)
        update                  = training(w, mu, sigma, eta)
        loglikelihood_          = loglikelihood(x, mu, sigma, w)
        xs.append(x)
        props.append(prop)
        updates.append(update)
        loglikelihoods.append(loglikelihood_)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sum_loglikelihood = 0
        for i, (train, update, loglikelihood_) in enumerate(zip(train_X, updates, loglikelihoods)):
            p_loglikelihood_ = -np.inf
            feed_dict = {xs[i]: train}
            print "target: {}".format(i)
            while sess.run(loglikelihood_, feed_dict)-p_loglikelihood_ > sess.run(loglikelihood_, feed_dict)*0.001 or p_loglikelihood_ < 0:
                p_loglikelihood_ = sess.run(loglikelihood_, feed_dict)
                print "loglikelihood: {}".format(p_loglikelihood_)
                sess.run(update,
                        feed_dict=feed_dict
                        )
            sum_loglikelihood += p_loglikelihood_
        predicts = []
        for i, prop in enumerate(props):
            feed_dict = {xs[i]: test_X}
            predicts.append(sess.run(prop, feed_dict))
        predicts = np.array(predicts)
        pred_y   = predicts.argmax(axis=0)
    print "f1_score: {}".format(f1_score(test_y, pred_y, average='macro'))
    print "accuracy: {}".format(accuracy_score(test_y, pred_y))











