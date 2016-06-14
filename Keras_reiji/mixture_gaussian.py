from __future__ import division
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('mixture', 3, "Number of mixture gaussian")
flags.DEFINE_integer('dimension', 784, "Number of Dimension")

mnist = fetch_mldata('MNIST original')
train_X = mnist.data.astype(np.float64)/255.
train_y = mnist.target.astype(np.int32)
train_X = [train_X[train_y == i] for i in range(10)]

estimate_mu    = []
estimate_w     = []
estimate_sugma = []

def gaussian(x, mu, sigma):
    A = 1./((2*np.pi*sigma**2)**(FLAGS.dimension/2))
    phi = tf.exp(-tf.reduce_sum(tf.batch_matmul(tf.expand_dims(tf.expand_dims(x, 1)-mu, 2),
                                                tf.expand_dims(tf.expand_dims(x, 1)-mu, 3)
                                                ), reduction_indices=[2, 3]
                                )
                    /(2*sigma**2)
                )
    return A*phi

def loglikelihood(x, mu, sigma, w, eps):
    return tf.reduce_sum(tf.log(tf.reduce_sum(w*gaussian(x, mu, sigma)+eps, reduction_indices=1)), reduction_indices=0)

x     = tf.placeholder(dtype=tf.float64, shape=(None, FLAGS.dimension))

w     = tf.Variable(tf.ones(shape=(FLAGS.mixture,), dtype=tf.float64)/FLAGS.mixture)
init  = np.concatenate((np.mean(train_X[0], axis=0, keepdims=True)+0.02, np.mean(train_X[0], axis=0, keepdims=True)*0.79, np.mean(train_X[0], axis=0, keepdims=True)), axis=0)
#mu = tf.Variable(init)
mu    = tf.Variable(tf.random_normal(shape=(FLAGS.mixture, FLAGS.dimension), mean=0.5, stddev=0.5, dtype=tf.float64))
sigma = tf.Variable(tf.ones(shape=(FLAGS.mixture,), dtype=tf.float64)*0.5)

eps = tf.constant(0.000000001, dtype=tf.float64)

_eta  = w*gaussian(x, mu, sigma)+eps
eta   = _eta/tf.expand_dims(tf.reduce_sum(_eta, reduction_indices=1), 1)
x_mu  = tf.batch_matmul(tf.expand_dims(tf.expand_dims(x, 1)-mu, 2), tf.expand_dims(tf.expand_dims(x, 1)-mu, 3))

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

loglikelihood = loglikelihood(x, mu, sigma, w, eps)


feed_dict = {x: train_X[0]}
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for _ in range(20):
        print "loglikelihood: ", sess.run(loglikelihood,
                feed_dict=feed_dict
                )
        #print "w: ", sess.run(w,
        #        feed_dict=feed_dict
        #        )
        m = sess.run(mu)
        #print m[m>0]
        #print "sigma: ", sess.run(sigma)
        print "_eta: ", sess.run(_eta,
                feed_dict=feed_dict
                )
        #_Eta = sess.run(_eta,
        #        feed_dict=feed_dict
        #        )
        #print "eta: ", sess.run(eta,
        #        feed_dict=feed_dict
        #        )
        #Eta = sess.run(eta,
        #        feed_dict=feed_dict
        #        )
        #print "x-mu: ", sess.run(x_mu,
        #        feed_dict=feed_dict
        #        )
        #print "new_w: ", sess.run(new_w,
        #        feed_dict=feed_dict
        #        )
        #print "new_mu: ", sess.run(new_mu,
        #        feed_dict=feed_dict
        #        )
        #print "new_sigma: ", sess.run(new_sigma,
        #        feed_dict=feed_dict
        #        )
        #_etA = 1./np.sum(0.3*_Eta, axis=1)
        #print "_Eta max: ", _Eta.max()
        #print "_Eta min: ", _Eta.min()
        #print "_etA max: ", _etA.max()
        #print "_etA min: ", _etA.min()
        #e = _Eta*_etA[:, np.newaxis]
        #print "nan: ", e[np.isnan(e)]
        #print "e max: ", e.max()

        sess.run(update,
                feed_dict=feed_dict
                )





