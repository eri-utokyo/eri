from __future__ import division
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('mixture', 3, "Number of mixture gaussian")
flags.DEFINE_integer('dimension', 784, "Number of Dimension")
flags.DEFINE_bool('pca', False, "Use PCA in preprocessing")

def gaussian(x, mu, sigma):
    """
    tf.expand_dims(x, 1): (len(train), 1, dimension)
    tf.expand_dims(x, 1)-mu: (len(train), mixture, dimension)
    x = tf.expand_dims(tf.expand_dims(x, 1)-mu, 2): (len(train), mixture, 1, dimension)
    y = tf.expand_dims(tf.expand_dims(x, 1)-mu, 3): (len(train), mixture, dimension, 1)
    input = tf.batch_matmul(x, y): (len(train), mixture, 1, 1)
    tf.reduce_sum(input, reduction_indices=[2, 3]): (len(train), mixture)
    """
    A = 1./((2*np.pi*(sigma**2))**(FLAGS.dimension/2))
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
    w     = tf.Variable(np.arange(1, FLAGS.mixture+1)/np.sum(np.arange(1, FLAGS.mixture+1)))
    mu    = tf.Variable(mu_init)
    sigma = tf.Variable(np.arange(1, FLAGS.mixture+1)/np.sum(np.arange(1, FLAGS.mixture+1)))
    eps   = tf.constant(1e-6, dtype=tf.float64)
    _eta  = w*gaussian(x, mu, sigma)+eps
    eta   = _eta/tf.expand_dims(tf.reduce_sum(_eta, reduction_indices=1), 1)
    prob  = tf.reduce_sum(_eta, reduction_indices=1)
    return w, mu, sigma, eta, _eta,  prob

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
    eps   = tf.constant(1e-6, dtype=tf.float64)
    return tf.reduce_sum(tf.log(tf.reduce_sum(w*gaussian(x, mu, sigma)+eps, reduction_indices=1)), reduction_indices=0)

def mean_concat(train, mixture):
    if mixture == 1:
        mu_init = np.mean(train, axis=0, keepdims=True)
        return mu_init*np.random.normal(1, 1e-1, size=mu_init.shape)
    mu_init = np.concatenate((np.mean(train, axis=0, keepdims=True), np.mean(train, axis=0, keepdims=True)))
    for _ in range(1, mixture-1):
        mu_init = np.concatenate((mu_init, np.mean(train, axis=0, keepdims=True)))
    return mu_init*np.random.normal(1, 1e-1, size=mu_init.shape)


if __name__ == '__main__':
    mnist   = fetch_mldata('MNIST original')
    train_X = mnist.data.astype(np.float64)/255.
    train_y = mnist.target.astype(np.int32)
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2)
    train_X = [train_X[train_y == i] for i in range(10)]
    if FLAGS.pca:
        X_train = []
        for train in train_X:
            pca = PCA(FLAGS.dimension)
            X_train.append(pca.fit_transform(train))
        train_X = X_train

    xs             = []
    probs          = []
    updates        = []
    loglikelihoods = []
    mus            = []
    sigmas         = []
    etas           = []
    _etas          = []
    ws             = []
    gauses         = []

    for i, train in enumerate(train_X):
        mu_init                       = mean_concat(train, FLAGS.mixture)
        x                             = tf.placeholder(dtype=tf.float64, shape=(None, FLAGS.dimension))
        w, mu, sigma, eta, _eta, prob = inference(x, mu_init)
        gaus                          = gaussian(x, mu, sigma)
        update                        = training(w, mu, sigma, eta)
        loglikelihood_                = loglikelihood(x, mu, sigma, w)
        xs.append(x)
        probs.append(prob)
        updates.append(update)
        loglikelihoods.append(loglikelihood_)
        mus.append(mu)
        sigmas.append(sigma)
        etas.append(eta)
        _etas.append(_eta)
        ws.append(w)
        gauses.append(gaus)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sum_loglikelihood = 0
        for i, (train, update, loglikelihood_) in enumerate(zip(train_X, updates, loglikelihoods)):
            p_loglikelihood_ = -np.inf
            feed_dict = {xs[i]: train}
            print "target: {}".format(i)
            print "sigma: {}".format(sess.run(sigmas[i], feed_dict))
            #print "w: {}".format(sess.run(ws[i], feed_dict))
            #print "_eta: {}".format(sess.run(_etas[i], feed_dict))
            #print "eta: {}".format(sess.run(etas[i], feed_dict))
            #print "gaus: {}".format(sess.run(gauses[i], feed_dict))
            while sess.run(loglikelihood_, feed_dict)-p_loglikelihood_ > sess.run(loglikelihood_, feed_dict)*0.01:
                p_loglikelihood_ = sess.run(loglikelihood_, feed_dict)
                print "loglikelihood_: {}".format(p_loglikelihood_)
                sess.run(update,
                        feed_dict=feed_dict
                        )
                print "sigma: {}".format(sess.run(sigmas[i], feed_dict))
                #print "w: {}".format(sess.run(ws[i], feed_dict))
                #print "_eta: {}".format(sess.run(_etas[i], feed_dict))
                #print "eta: {}".format(sess.run(etas[i], feed_dict))
                #print "gaus: {}".format(sess.run(gauses[i], feed_dict))
            sum_loglikelihood += p_loglikelihood_
        predicts = []
        if FLAGS.pca:
            pca = PCA(FLAGS.dimension)
            test_X = pca.fit_transform(test_X)
        for i, prob in enumerate(probs):
            feed_dict = {xs[i]: test_X}
            predicts.append(sess.run(prob, feed_dict))
        predicts = np.array(predicts)
        print predicts[:, 0:5]
        pred_y   = predicts.argmax(axis=0)
    print "f1_score: {}".format(f1_score(test_y, pred_y, average='macro'))
    print "accuracy: {}".format(accuracy_score(test_y, pred_y))











