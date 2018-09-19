
#-------------------------------------------------------------------------------------------------#
# This file provide an example of making inference using Mixture Density Network of dSph galaxies #
#-------------------------------------------------------------------------------------------------#

import numpy as np
import random
import tensorflow as tf
import time
import os
import utility as ut2
import tensorflow.contrib.distributions as tfd


def weights(shape):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        return tf.Variable(initial)

def bias(shape):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

# The transformation that standardize each parameter to within [0,1] using the prior of each
# parameter. Ylist is a list of the real value parameters.
def norm_Y(Ylist):
    a,       d,   e,   Ec,  rlim, b,   q,    Jb,   Menc,       rs, al, be, ga = Ylist
    Y = np.array( [a/7.1, (d-1)/-10., e/5., (Ec-.05)/.51, (rlim+.5)/2., (b+5)/10., (q-.5)/10.,
                 (Jb-.01)/.5, (Menc-6)/3., (rs+1)/1.5, ga/1.5] )
    return Y


# Reading the mock data and Fornax data for Inference; 
# max_bins is number of 2D grid in KDE density, in this case it's max_bins=30. 
def get_test(max_bins):

    #For example, data is located below.
    fdir = './cvae/mock/data_kde/'
    f1 = fdir+'mock_sfw_b2_ga0_100k_05std_2std_RE_30grid_masked'
    f2 = fdir+'mock_sfw_b2_ga1_100k_05std_2std_30grid_masked'
    f0 = fdir+'for_published_data_30grid_masked'

    # read and transform mock data to make inference
    datasize = max_bins * max_bins
    Xflat0 = np.reshape( np.loadtxt(f0, delimiter=' ', unpack=True, usecols=[0]), [-1, datasize])
    Xflat1 = np.reshape( np.loadtxt(f1, delimiter=' ', unpack=True, usecols=[0]), [-1, datasize])
    Xflat2 = np.reshape( np.loadtxt(f2, delimiter=' ', unpack=True, usecols=[0]), [-1, datasize])

    # parameter truth
    Y1 = [ norm_Y([2.0, -5.3, 4.5, 0.16,     0, 2, 6.9, 0.086, 8.0513, -.158365, 1., 3., 0.]) ]
    Y2 = [ norm_Y([2.0, -5.3, 2.5, 0.16,  .176, 2, 6.9, 0.086, 7.8234, -.158365, 1., 3., 1.]) ]
    Y0 = [ np.array([0]*11) ] # real data, we don't know the truth

    return Xflat0, Xflat1, Xflat2, Y0, Y1, Y2, datasize

#test data to make inference
maxbins = 30
Xflat0, Xflat1, Xflat2, Y0, Y1, Y2, datasize = get_test(maxbins)



#-------- setting ------------------------------------
#dset = 'p11g30-10M'
ckpt_epoch = 5       # to save training parameters every 'ckpt_epoch' epochs
sample_epoch = 2     # save the inferred posteriors every 'sample_epoch' epoches
inputsize = datasize # size of the input X
n_y = 11             # size of the label (number of parameters) 
diag = False         # covariance matrix for the mixture model is NOT diagonal

num_epoch = 1000  #number of epoch to train
traintf = ['p11_train_30grid_masked.tfrecords'] # training tfrecord file
validtf = ['p11_valid_30grid_masked.tfrecords'] # validation tfrecord file
num_train = 1000000  # number of training data
num_valid = 100000   # number of validation data


#------- hyperparameters -----------------------------
batch_size = 50  # training batch size
n_mixture = 60   # number of Gaussians to approximate the posterior
n_z = n_y * n_mixture
n_hidden_1 = 200  # number of neurons in first layer
n_hidden_2 = 200  # number of neurons in second layer


# learning rate setting
global_step = tf.Variable(0, trainable=False)
lr = 3e-5    # initial learning rate
decay = .99  # decay rate
decay_step = num_train/batch_size
slr = tf.train.exponential_decay(lr, global_step, decay_step, decay, staircase=True)

# save the setting info in the file name
info = "p%dkde_grid%d_%dby%d_bsize%d_decay%dlr%.e_nmix%d_%dMtrain" % (n_y, maxbins,
                n_hidden_1,n_hidden_2, batch_size, decay*100, lr, n_mixture, num_train/1000000 )

# directory for inferred posterior
fypred = './ypred_samples/'+ info +'/'
if not os.path.exists(fypred):
    os.makedirs(fypred)

# directory for checkpoints
checkpoint_path = "./checkpoint/p11kde/"+ info +"/"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
ckptname = '_' + info



#----------------- beginning of the neural network.
with tf.name_scope('input'):
    x = tf.placeholder("float32", shape=[None, inputsize], name='x')
    y_true = tf.placeholder("float32", shape=[None, n_y], name='y')
    inputs = x 

# First hidden layer
with tf.name_scope("layer1"):
    W_fc1 = weights([inputsize, n_hidden_1])
    b_fc1 = bias([n_hidden_1])
    h_1   = tf.nn.softplus(tf.matmul(inputs, W_fc1) + b_fc1)

# Second hidden layer
with tf.name_scope("layer2"):
    W_fc2 = weights([n_hidden_1, n_hidden_2])
    b_fc2 = bias([n_hidden_2])
    h_2   = tf.nn.softplus(tf.matmul(h_1, W_fc2) + b_fc2)


# Output layer (courtesy of François Lanusse)
# latent mean and ls2(log_sigma_square), and the mixture weights, use to sample z
size_sigma = n_y if diag else n_y*(n_y+1)//2 
with tf.name_scope("mu-logvar-weight-layer"):
    W_mu = weights([n_hidden_2, n_z])
    b_mu = bias([n_z])
    out_mu = tf.add(tf.matmul(h_2, W_mu), b_mu)
    out_mu = tf.reshape(out_mu, (-1, n_mixture, n_y))

    W_sigma = weights([n_hidden_2, size_sigma*n_mixture])
    b_sigma = bias([size_sigma*n_mixture])
    out_sigma = tf.add(tf.matmul(h_2, W_sigma), b_sigma)
    #out_sigma = tf.clip_by_value(out_sigma, -10, 10) # For stability
    out_sigma = tf.reshape(out_sigma, (-1, n_mixture, size_sigma))

    W_p = weights([n_hidden_2, n_mixture])
    b_p = bias([n_mixture])
    out_p = tf.add(tf.matmul(h_2, W_p), b_p)
    out_p = tf.nn.softmax(tf.reshape(out_p, (-1, n_mixture)))

    if diag:
        sigma_mat = tf.nn.softplus(out_sigma)
        gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=out_p),
                      components_distribution=tfd.MultivariateNormalDiag(loc=out_mu,
                                                        scale_diag=sigma_mat))
    else:
        sigma_mat = tfd.matrix_diag_transform(tfd.fill_triangular(out_sigma), transform=tf.nn.softplus)
        gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=out_p),
                     components_distribution=tfd.MultivariateNormalTriL(loc=out_mu,
                                                        scale_tril=sigma_mat))

    predictions = {'mu': out_mu, 'sigma': sigma_mat, 'p':out_p}


# ------------------- network loss and optimizer -----------------------------------------
with tf.name_scope('Cost'):
    loss = - tf.reduce_mean(gmm.log_prob(y_true),axis=0)
    tf.losses.add_loss(loss)
    cost = tf.losses.get_total_loss() #the loss function is negative of the log-likelihood. 
with tf.name_scope('SGD'):
    optimizer = tf.train.AdamOptimizer(learning_rate=slr).minimize(cost, global_step=global_step)
# --------------------------------------------------------------------------------------------


# From input 'x0', posterior approximated by density mixture is calculated with the network, 
# samples are drawn from the estimated posterior.
def sample_y(x0):
    h_1_s  = tf.nn.softplus(tf.matmul(x0, W_fc1) + b_fc1)
    h_2_s  = tf.nn.softplus(tf.matmul(h_1_s, W_fc2) + b_fc2)

    out_mu1  = tf.add(tf.matmul(h_2_s, W_mu), b_mu)
    out_mu1 = tf.reshape(out_mu1, (-1, n_mixture, n_y))

    out_sigma1 = tf.add(tf.matmul(h_2_s, W_sigma), b_sigma)
    out_sigma1 = tf.reshape(out_sigma1, (-1, n_mixture, size_sigma))

    out_p1 = tf.add(tf.matmul(h_2_s, W_p), b_p)
    out_p1 = tf.nn.softmax(tf.reshape(out_p1, (-1, n_mixture+0)))

    if diag == True:
        sigma_mat1 = tf.nn.softplus(out_sigma1)
        gmm1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=out_p1),
                      components_distribution=tfd.MultivariateNormalDiag(loc=out_mu1,
                                                  scale_diag=sigma_mat1))
    else:
        sigma_mat1 = tfd.matrix_diag_transform(tfd.fill_triangular(out_sigma1), transform=tf.nn.softplus)
        gmm1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=out_p1),
                     components_distribution=tfd.MultivariateNormalTriL(loc=out_mu1,
                                                  scale_tril=sigma_mat1))
    y3 = gmm1.sample()
    return y3

# posterior samples (corresponds to parameters) of the input x 
y_samples = sample_y(x)

# Saved the 10000 posteriors drawn to the file 'fname', for a given galaxy 'x_test'. 
# 'y_test' is appended to the first line, in case it's mock data and the truth is known.
def save_samples(fname, x_test, y_test): 
    samples = [y_test]
    for i in xrange(10000):
        zrand = np.random.randn(1, n_z)
        y_pred = sess.run( [y_samples], feed_dict={x: x_test} )
        samples.append(y_pred[0])

    samples = np.reshape( np.array(samples), (10001,n_y))
    np.savetxt(fypred + fname, np.array(samples), fmt='%1.5f')
    return 0


#------------------------------ Training --------------------------------------------------------
# read the training and validation tfrecord files
timages, tlabels = ut2.read_data(traintf, datasize, batch_size, num_train/20, num_train/50, n_y)
vimages, vlabels = ut2.read_data(validtf, datasize, batch_size, num_valid/20, num_valid/50, n_y)
total_train_batch = num_train / batch_size
total_valid_batch = num_valid / batch_size


init = tf.global_variables_initializer()
with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        for epoch in xrange(num_epoch):
            t0 = time.time()
            current_lr = sess.run(slr) # current learning rate

            for i in xrange( total_train_batch ): #train over the training data batch
                # read and reshape the data to feed into the optimizer
                tbatch_xs, tbatch_ys = sess.run([timages, tlabels])
                shape = int(np.prod(  np.shape(tbatch_xs)[1:]  ))
                tbatch_xs = np.reshape(tbatch_xs, [-1, shape])

                _ = sess.run([optimizer], feed_dict = {x: tbatch_xs, y_true: tbatch_ys})

            
            # to check the average training loss
            avg_train_cost = 0
            for i in xrange( total_train_batch ):
                tbatch_xs, tbatch_ys = sess.run([timages, tlabels])
                shape = int(np.prod(  np.shape(tbatch_xs)[1:]  ))
                tbatch_xs = np.reshape(tbatch_xs, [-1, shape])

                c = sess.run(cost, feed_dict = {x: tbatch_xs, y_true: tbatch_ys})
                avg_train_cost += c / total_train_batch


            # to check the average validation loss
            avg_valid_cost = 0.
            for j in xrange( total_valid_batch*a+b ):
                vbatch_xs, vbatch_ys = sess.run([vimages, vlabels])
                shape = int(np.prod(  np.shape(vbatch_xs)[1:]  ))
                vbatch_xs = np.reshape(vbatch_xs, [-1, shape])

                vloss = sess.run( cost, feed_dict={x: vbatch_xs, y_true: vbatch_ys})
                avg_valid_cost += vloss / (total_valid_batch)

    
            # print current training state 
            log = str(epoch+1) + ' ' + str("{:.5f}".format(avg_train_cost)) + \
                ' ' + str("{:.5f}".format(avg_valid_cost)) + \
                + ' '+ str(current_lr) + ' '+ str(time.time()-t0)
            print log


            # to save check point, in case to restart training
            if (epoch+1) % ckpt_epoch == 0:
                checkpoint_name = checkpoint_path  + ckptname + str(epoch+1)+'.ckpt'
                save_path = saver.save(sess, checkpoint_name)



            # input a galaxy to draw posteriors (estimate parameter distribution)
            if (epoch+1) % sample_epoch == 0:
                Xflat0, Xflat1, Xflat2, Y0, Y1, Y2, datasize = get_test(maxbins)
                name = "_nz%d_bsize%d_maxbins%d_lr%d_epoch%d.out" % (n_z,
                                                batch_size, maxbins, lr*1000, epoch+1)
                fname0 = dset + "_f0" + name # fornax data 
                fname1 = dset + "_f1" + name # core mock data
                fname2 = dset + "_f2" + name # cusp mock data

                _ = save_samples(fname0, Xflat0, Y0)
                _ = save_samples(fname1, Xflat1, Y1)
                _ = save_samples(fname2, Xflat2, Y2)

        # finalise 
        coord.request_stop()
        coord.join(threads)
























