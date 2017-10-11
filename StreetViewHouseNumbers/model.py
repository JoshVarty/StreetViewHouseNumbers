import os
import numpy as np
import matplotlib.pyplot as plt
import data_loader
import tensorflow as tf
from sklearn.model_selection import train_test_split

image_size = 32
num_channels = 1
pixel_depth = 255
num_labels = 5
num_digits = 10
patch_size_3 = 3
depth = 16

train_data, train_labels, valid_data, valid_labels = data_loader.load_data()

print("Train data", train_data.shape)
print("Train labels", train_labels.shape)
print("Valid data", valid_data.shape)
print("Valid labels", valid_labels.shape)

def accuracy(labels, predictions):
    return (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

def TrainConvNet():

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
        input = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
        labels = tf.placeholder(tf.int32, shape=(None))
        #keep_prob = tf.placeholder(tf.float32)

        #Conv->Relu->Conv->Relu->Pool
        w_conv1 = weight_layer("w_conv1", [patch_size_3, patch_size_3, num_channels, depth])
        b_conv1 = bias_variable("b_conv1", [depth])
        h_conv1 = conv2d_relu(input, w_conv1, b_conv1)
        w_conv2 = weight_layer("w_conv2", [patch_size_3, patch_size_3, depth, depth])
        b_conv2 = bias_variable("b_conv2", [depth])
        h_conv2 = conv2d_relu(h_conv1, w_conv2, b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        #Conv->Relu->Conv->Relu->Pool
        w_conv3 = weight_layer("w_conv3", [patch_size_3, patch_size_3, depth, depth * 2])
        b_conv3 = bias_variable("b_conv3", [depth * 2])
        h_conv3 = conv2d_relu(h_pool2, w_conv3, b_conv3)
        w_conv4 = weight_layer("w_conv4", [patch_size_3, patch_size_3, depth * 2, depth * 4])
        b_conv4 = bias_variable("b_conv4", [depth * 4])
        h_conv4 = conv2d_relu(h_conv3, w_conv4, b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)

        #Conv->Relu->Conv->Relu->Conv->Relu->Pool
        w_conv5 = weight_layer("w_conv5", [patch_size_3, patch_size_3, depth * 4, depth * 4])
        b_conv5 = bias_variable("b_conv5", [depth * 4])
        h_conv5 = conv2d_relu(h_pool4, w_conv5, b_conv5)
        w_conv6 = weight_layer("w_conv6", [patch_size_3, patch_size_3, depth * 4, depth * 4])
        b_conv6 = bias_variable("b_conv6", [depth * 4])
        h_conv6 = conv2d_relu(h_conv5, w_conv6, b_conv6)
        w_conv7 = weight_layer("w_conv7", [patch_size_3, patch_size_3, depth * 4, depth * 8])
        b_conv7 = bias_variable("b_conv7", [depth * 8])
        h_conv7 = conv2d_relu(h_conv6, w_conv7, b_conv7)
        h_pool7 = max_pool_2x2(h_conv7)
        
        #Conv->Relu->Conv->Relu->Conv->Relu->Pool
        w_conv8 = weight_layer("w_conv8", [patch_size_3, patch_size_3, depth * 8, depth * 8])
        b_conv8 = bias_variable("b_conv8", [depth * 8])
        h_conv8 = conv2d_relu(h_pool7, w_conv8, b_conv8)
        w_conv9 = weight_layer("w_conv9", [patch_size_3, patch_size_3, depth * 8, depth * 8])
        b_conv9 = bias_variable("b_conv9", [depth * 8])
        h_conv9 = conv2d_relu(h_conv8, w_conv9, b_conv9)
        w_conv10 = weight_layer("w_conv10", [patch_size_3, patch_size_3, depth * 8, depth * 16])
        b_conv10 = bias_variable("b_conv10", [depth * 16])
        h_conv10 = conv2d_relu(h_conv9, w_conv10, b_conv10)
        h_pool10 = max_pool_2x2(h_conv10)

        #Dropout -> Fully Connected -> Dropout -> Fully Connected
        drop_1 = tf.nn.dropout(h_pool10, 1.0)
        shape = drop_1.get_shape().as_list()
        reshape = tf.reshape(drop_1, [-1, shape[1] * shape[2] * shape[3]])

        fc = 1024
        hl = 4096
        w_fc = weight_layer("w_fc", [fc, hl])
        b_fc = bias_variable("b_fc", [hl])
        h_fc = tf.matmul(reshape, w_fc) + b_fc

        #w_l = weight_layer("w_l", [hl, 5])
        #b_l = weight_layer("b_l", [5])
        #z_l = tf.matmul(h_fc, w_l) + b_l

        w_s_1 = weight_layer("z_s_1", [hl, num_digits])
        b_s_1 = bias_variable("b_s_1", [num_digits])
        z_s_1 = tf.matmul(h_fc, w_s_1) + b_s_1
        
        #w_s_2 = weight_layer("z_s_2", [hl, num_digits])
        #b_s_2 = bias_variable("b_s_2", [num_digits])
        #z_s_2 = tf.matmul(h_fc, w_s_2) + b_s_2
        
        #w_s_3 = weight_layer("z_s_3", [hl, num_digits])
        #b_s_3 = bias_variable("b_s_3", [num_digits])
        #z_s_3 = tf.matmul(h_fc, w_s_3) + b_s_3

        #w_s_4 = weight_layer("z_s_4", [hl, num_digits])
        #b_s_4 = bias_variable("b_s_4", [num_digits])
        #z_s_4 = tf.matmul(h_fc, w_s_4) + b_s_4

        #w_s_5 = weight_layer("z_s_5", [hl, num_digits])
        #b_s_5 = bias_variable("b_s_5", [num_digits])
        #z_s_5 = tf.matmul(h_fc, w_s_5) + b_s_5

        #labels1 = tf.slice(labels, [0, 0], [-1, 1])
        #labels2 = tf.slice(labels, [0, 1], [-1, 1])
        #labels3 = tf.slice(labels, [0, 2], [-1, 1])
        #labels4 = tf.slice(labels, [0, 3], [-1, 1])
        #labels5 = tf.slice(labels, [0, 4], [-1, 1])

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=z_s_1))
        #cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels2, logits=z_s_2))
        #cost3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels3, logits=z_s_3))
        #cost4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels4, logits=z_s_4))
        #cost5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels5, logits=z_s_5))

        #total_cost = cost1 + cost2 + cost3 + cost4 + cost5

        #train_prediction = tf.stack([
        #    tf.nn.softmax(z_s_1),
        #    tf.nn.softmax(z_s_2),
        #    tf.nn.softmax(z_s_3),
        #    tf.nn.softmax(z_s_4),
        #    tf.nn.softmax(z_s_5)
        #    ], axis=1)

        train_prediction = tf.nn.softmax(z_s_1)

        optimizer = tf.train.AdamOptimizer(0.0000001).minimize(cost)

        with tf.Session(graph=graph) as session:
            num_steps = 10000
            batch_size = 32
            tf.global_variables_initializer().run()
            print("Initialized")

            for step in range(num_steps):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_data[offset:(offset + batch_size), :, :]
                batch_labels = np.squeeze(train_labels[offset:(offset + batch_size), :])

                feed_dict = {input : batch_data, labels : batch_labels} 

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

                        feed_dict = {input : v_batch_data, labels : v_batch_labels}
                        l, predictions = session.run([cost, train_prediction], feed_dict=feed_dict)
                        v_preds[v_offset: v_offset + predictions.shape[0],:] = predictions

                    #If we missed any validation images at the end, process them now
                    if v_steps * v_batch_size < valid_data.shape[0]:
                        v_offset = (v_steps * v_batch_size) 
                        v_batch_data = valid_data[v_offset:valid_data.shape[0] , :, :, :]
                        v_batch_labels = np.squeeze(valid_labels[v_offset:valid_data.shape[0],:])

                        feed_dict = {input : v_batch_data, labels : v_batch_labels}
                        l, predictions, = session.run([total_cost, train_prediction], feed_dict=feed_dict)
                        v_preds[v_offset: v_offset + predictions.shape[0],:] = predictions

                    print('Valid accuracy: %.1f%%' % accuracy(np.squeeze(valid_labels), v_preds))
                else:
                    _, l, predictions, = session.run([optimizer, cost, train_prediction], feed_dict=feed_dict)


TrainConvNet()
