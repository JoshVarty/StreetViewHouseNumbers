import os
import numpy as np
#import matplotlib.pyplot as plt
import data_loader
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.model_selection import train_test_split

image_size = 32
cropped_size = 28
num_channels = 1
pixel_depth = 255
num_labels = 5
num_digits = 10
depth = 32

patch_size_1 = 1
patch_size_3 = 3
patch_size_5 = 5
patch_size_7 = 7

train_data, train_labels, valid_data, valid_labels = data_loader.load_data()

print("Train data", train_data.shape)
print("Train labels", train_labels.shape)
print("Valid data", valid_data.shape)
print("Valid labels", valid_labels.shape)

def accuracy(labels, predictions):
    return (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

def TrainConvNet():

    def LecunLCN(X, image_shape, threshold=1e-4, radius=7, use_divisor=True):
        """Local Contrast Normalization"""
        """[http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf]"""

        # Get Gaussian filter
        filter_shape = (radius, radius, image_shape[3], 1)

        #self.filters = theano.shared(self.gaussian_filter(filter_shape), borrow=True)
        filters = gaussian_filter(filter_shape)
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        # Compute the Guassian weighted average by means of convolution
        convout = tf.nn.conv2d(X, filters, [1,1,1,1], 'SAME')

        # Subtractive step
        mid = int(np.floor(filter_shape[1] / 2.))

        # Make filter dimension broadcastable and subtract
        centered_X = tf.subtract(X, convout)

        # Boolean marks whether or not to perform divisive step
        if use_divisor:
            # Note that the local variances can be computed by using the centered_X
            # tensor. If we convolve this with the mean filter, that should give us
            # the variance at each point. We simply take the square root to get our
            # denominator

            # Compute variances
            sum_sqr_XX = tf.nn.conv2d(tf.square(centered_X), filters, [1,1,1,1], 'SAME')

            # Take square root to get local standard deviation
            denom = tf.sqrt(sum_sqr_XX)

            per_img_mean = tf.reduce_mean(denom)
            divisor = tf.maximum(per_img_mean, denom)
            # Divisise step
            new_X = tf.truediv(centered_X, tf.maximum(divisor, threshold))
        else:
            new_X = centered_X

        return new_X

    def gaussian_filter(kernel_shape):
        x = np.zeros(kernel_shape, dtype = float)
        mid = np.floor(kernel_shape[0] / 2.)
        
        for kernel_idx in range(0, kernel_shape[2]):
            for i in range(0, kernel_shape[0]):
                for j in range(0, kernel_shape[1]):
                    x[i, j, kernel_idx, 0] = gauss(i - mid, j - mid)
        
        return tf.convert_to_tensor(x / np.sum(x), dtype=tf.float32)

    def gauss(x, y, sigma=3.0):
        Z = 2 * np.pi * sigma ** 2
        return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    def weight_layer(name, shape):
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(name, shape):
          return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

    def conv2d_relu(input, weights, bias):
        return tf.nn.relu(tf.nn.conv2d(input, weights, [1,1,1,1], padding="SAME") + bias)

    def max_pool_2x2(input):    
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    graph = tf.Graph()
    with graph.as_default():
        input = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels), name="input")
        labels = tf.placeholder(tf.int32, shape=(None), name="labels")
        keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")
        learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")

        LCN = LecunLCN(input, (None, image_size, image_size, num_channels))

        with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):

            net = slim.conv2d(LCN, 64, [3,3])
            net = slim.conv2d(net, 64, [3,3])

            #Inception Module 1
            branch_0 = slim.conv2d(net, 64, [1, 1])
            branch_1 = slim.conv2d(net, 96, [1, 1])
            branch_1 = slim.conv2d(branch_1, 128, [3, 3])
            branch_2 = slim.conv2d(net, 16, [1, 1])
            branch_2 = slim.conv2d(branch_2, 32, [3, 3])
            branch_3 = slim.max_pool2d(net, [5, 5])
            branch_3 = slim.conv2d(branch_3, 32, [1, 1])
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            #Inception Module 2
            branch_0 = slim.conv2d(net, 128, [1, 1])
            branch_1 = slim.conv2d(net, 128, [1, 1])
            branch_1 = slim.conv2d(branch_1, 192, [3, 3])
            branch_2 = slim.conv2d(net, 32, [1, 1])
            branch_2 = slim.conv2d(branch_2, 96, [5, 5])
            branch_3 = slim.max_pool2d(net, [3, 3])
            branch_3 = slim.conv2d(branch_3, 64, [1, 1])
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            #MaxPool
            net = slim.max_pool2d(net, [3,3], stride=2)

            #Inception Module 3
            branch_0 = slim.conv2d(net, 192, [1, 1])
            branch_1 = slim.conv2d(net, 96, [1, 1])
            branch_1 = slim.conv2d(branch_1, 208, [3, 3])
            branch_2 = slim.conv2d(net, 16, [1, 1])
            branch_2 = slim.conv2d(branch_2, 48, [5, 5])
            branch_3 = slim.max_pool2d(net, [3, 3])
            branch_3 = slim.conv2d(branch_3, 64, [1, 1])
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            #Inception Module 4
            branch_0 = slim.conv2d(net, 160, [1, 1])
            branch_1 = slim.conv2d(net, 112, [1, 1])
            branch_1 = slim.conv2d(branch_1, 224, [3, 3])
            branch_2 = slim.conv2d(net, 24, [1, 1])
            branch_2 = slim.conv2d(branch_2, 64, [5, 5])
            branch_3 = slim.max_pool2d(net, [3, 3])
            branch_3 = slim.conv2d(branch_3, 64, [1, 1])
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

            #MaxPool and 1x1 Conv to reduce dimensions
            net = slim.max_pool2d(net, [3,3], stride=2)
            net = slim.conv2d(net, 128, [1,1])

            #Two Fully Connected
            shape = net.get_shape().as_list()
            reshape = tf.reshape(net, [-1, shape[1] * shape[2] * shape[3]])
            fc = slim.fully_connected(reshape, 1024)
            fc = slim.fully_connected(fc, 1024)
            logits = slim.fully_connected(fc, 10)

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        train_prediction = tf.nn.softmax(logits)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        with tf.Session(graph=graph) as session:
            writer = tf.summary.FileWriter("/tmp/svhn_single")
            writer.add_graph(session.graph)
            num_steps = 60000
            batch_size = 64
            tf.global_variables_initializer().run()
            print("Initialized")
            lr = 0.0001
            for step in range(num_steps):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_data[offset:(offset + batch_size), :, :]
                batch_labels = np.squeeze(train_labels[offset:(offset + batch_size), :])

                feed_dict = {input : batch_data, labels : batch_labels, keep_prob : 0.5, learning_rate: lr} 

                if step % 10000 == 0:
                    lr = lr / 2
                    print("Learning Rate: ", lr)

                if step % 500 == 0:
                    _, l, predictions, = session.run([optimizer, cost, train_prediction], feed_dict=feed_dict)
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy(batch_labels, predictions)) 
                    #Validation

                    v_steps = 10
                    v_batch_size = int(valid_data.shape[0] / v_steps)
                    v_preds = np.zeros((valid_labels.shape[0], num_digits))
                    for v_step in range(v_steps):
                        v_offset = (v_step * v_batch_size) 
                        v_batch_data = valid_data[v_offset:(v_offset + v_batch_size), :, :]
                        v_batch_labels = np.squeeze(valid_labels[v_offset:(v_offset + v_batch_size),:])

                        feed_dict = {input : v_batch_data, labels : v_batch_labels, keep_prob : 1.0, learning_rate: lr}
                        l, predictions = session.run([cost, train_prediction], feed_dict=feed_dict)
                        v_preds[v_offset: v_offset + predictions.shape[0],:] = predictions

                    #If we missed any validation images at the end, process them now
                    if v_steps * v_batch_size < valid_data.shape[0]:
                        v_offset = (v_steps * v_batch_size) 
                        v_batch_data = valid_data[v_offset:valid_data.shape[0] , :, :, :]
                        v_batch_labels = np.squeeze(valid_labels[v_offset:valid_data.shape[0],:])

                        feed_dict = {input : v_batch_data, labels : v_batch_labels, keep_prob : 1.0, learning_rate: lr}
                        l, predictions, = session.run([total_cost, train_prediction], feed_dict=feed_dict)
                        v_preds[v_offset: v_offset + predictions.shape[0],:] = predictions

                    print('Valid accuracy: %.1f%%' % accuracy(np.squeeze(valid_labels), v_preds))
                else:
                    _, l, predictions, = session.run([optimizer, cost, train_prediction], feed_dict=feed_dict)


TrainConvNet()

