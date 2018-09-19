
#--------------------------------------------------------------------------------------------#
# This file provide an example of making inference using Conditional Variational Autoencoder #
# of dSph galaxies.                                                                          #
#--------------------------------------------------------------------------------------------#


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
batch_size = 10   # training batch size
n_hidden_1 = 100  # number of neurons in first layer of encoder
n_hidden_2 = 100  # in second layer of encoder
n_hidden_1g = 100 # number of neurons in first layer of decoder
n_hidden_2g = 100 # in second layer of decoder
n_z = 20          # number of latent variables

# optional
lfac0=1       # the factor that multiply the reconstruction loss
lfac1=10000   # the new factor that multiply the reconstruction loss after a certain epoch, see below


# setup the learning rate and decay
global_step = tf.Variable(0, trainable=False)
lr = 1e-4    # initial learning rate
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



#----------------- beginning of the neural network-----------------------------
with tf.name_scope('input'):
    x = tf.placeholder("float32", shape=[None, inputsize], name='x')  # input array
    y_true = tf.placeholder("float32", shape=[None, n_out], name='y') # label
    inputs = tf.concat(axis=1, values=[x, y_true], name='input') 
    z1 = tf.placeholder("float32", shape=[None, n_z], name='z1') #latent z use to sample parameters y

# First hidden layer (encoder)
with tf.name_scope("layer1"):
    W_fc1 = weights([inputsize+n_out, n_hidden_1])
    b_fc1 = bias([n_hidden_1])
    h_1   = tf.nn.softplus(tf.matmul(inputs, W_fc1) + b_fc1)

# Second hidden layer (encoder)
with tf.name_scope("layer2"):
    W_fc2 = weights([n_hidden_1, n_hidden_2])
    b_fc2 = bias([n_hidden_2])
    h_2   = tf.nn.softplus(tf.matmul(h_1, W_fc2) + b_fc2)

#latent mean and ls2(log_sigma_square), use to sample z
with tf.name_scope("z-samples"):
    z_mu  = tf.add(tf.matmul(h_2, weights([n_hidden_2, n_z])), bias([n_z]))
    z_ls2 = tf.add(tf.matmul(h_2, weights([n_hidden_2, n_z])), bias([n_z]))
    eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32) # Adding a random number
    z = tf.add(z_mu, tf.multiply(tf.sqrt(tf.exp(z_ls2)), eps))        # The sampled z


with tf.name_scope('decode_input'):
    inputs_g = tf.concat(axis=1, values=[x, z]) if new else tf.concat(1, [x, z])

# Decoder hidden layer 1
with tf.name_scope("DC-layer1"):
    W_fc1_g = weights([n_z+inputsize, n_hidden_1g])
    b_fc1_g = bias([n_hidden_1g])
    h_1_g   = tf.nn.softplus(tf.matmul(inputs_g, W_fc1_g) + b_fc1_g)

#Decoder hidden layer 2
with tf.name_scope("DC-layer2"):
    W_fc2_g = weights([n_hidden_1g, n_hidden_2g])
    b_fc2_g = bias([n_hidden_2g])
    h_2_g   = tf.nn.softplus(tf.matmul(h_1_g, W_fc2_g) + b_fc2_g)

# reconstruct layer
W_x = weights([n_hidden_2g, n_y])
b_x = bias([n_y])
with tf.name_scope('y_pred'):
    Y_out = tf.matmul(h_2_g, W_x) + b_x  # the predicted parameters



# ------------------- total loss and optimizer -----------------------------------------
# lfac is the factor that multiply the reconstruction loss
lfac = tf.Variable(lfac0, trainable=False, dtype=tf.float32) 
with tf.name_scope('split_losses'):
    reconstr_loss0 = tf.reduce_sum(tf.squared_difference(Y_out, y_true), 1)
    reconstr_loss = tf.scalar_mul(lfac, reconstr_loss0)
    latent_loss   = -0.5 * tf.reduce_sum(1 + z_ls2 - tf.square(z_mu) - tf.exp(z_ls2), 1)

with tf.name_scope('Cost'):
    cost = tf.reduce_mean(reconstr_loss + latent_loss)
with tf.name_scope('SGD'):
    optimizer = tf.train.AdamOptimizer(learning_rate=slr).minimize(cost, global_step=global_step)
# --------------------------------------------------------------------------------------------


# to sample posterior through input variable x0, and latent variables z0 
def sample_y(x0,z0):
    inputs_s = tf.concat(axis=1, values=[x0, z0]) if new else tf.concat(1, [x0, z0])
    h_1_s    = tf.nn.softplus(tf.matmul(inputs_s, W_fc1_g) + b_fc1_g)
    h_2_s    = tf.nn.softplus(tf.matmul(h_1_s, W_fc2_g) + b_fc2_g)
    output_s = tf.matmul(h_2_s, W_x) + b_x
    return output_s

# posterior samples (corresponds to parameters) of the input x 
y_samples = sample_y(x, z1)

# Saved the 10000 posteriors drawn to the file 'fname', for a given galaxy 'x_test'. 
# 'y_test' is appended to the first line, in case it's mock data and the truth is known.
def save_samples(fname, x_test, y_test):
    samples = [y_test]
    for i in xrange(10000):
        zrand = np.random.randn(1, n_z) # latent variables are drawn from Gaussian distribution.
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
   
            # optional the factor that multiply the reconstruction loss changes from lfac0 to lfac1
            # after 50 epochs
            lfac2 = lfac0 if epoch < 50 else lfac1 

            for i in xrange( total_train_batch ): #train over the training data batch
                # read and reshape the data to feed into the optimizer
                tbatch_xs, tbatch_ys = sess.run([timages, tlabels])
                shape = int(np.prod(  np.shape(tbatch_xs)[1:]  ))
                tbatch_xs = np.reshape(tbatch_xs, [-1, shape])

                _ = sess.run([optimizer], feed_dict = {x: tbatch_xs, y_true: tbatch_ys, lfac:lfac2})

            
            # to check the average training loss
            avg_train_cost = 0
            for i in xrange( total_train_batch ):
                tbatch_xs, tbatch_ys = sess.run([timages, tlabels])
                shape = int(np.prod(  np.shape(tbatch_xs)[1:]  ))
                tbatch_xs = np.reshape(tbatch_xs, [-1, shape])

                c = sess.run(cost, feed_dict = {x: tbatch_xs, y_true: tbatch_ys, lfac:lfac2})
                avg_train_cost += c / total_train_batch


            # to check the average validation loss
            avg_valid_cost = 0.
            for j in xrange( total_valid_batch*a+b ):
                vbatch_xs, vbatch_ys = sess.run([vimages, vlabels])
                shape = int(np.prod(  np.shape(vbatch_xs)[1:]  ))
                vbatch_xs = np.reshape(vbatch_xs, [-1, shape])

                vloss = sess.run( cost, feed_dict={x: vbatch_xs, y_true: vbatch_ys, lfac:lfac2})
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
























