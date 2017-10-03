import numpy as np
import os
import data_loader
import image_helpers
from digit_struct import DigitStruct
import tensorflow as tf
from sklearn.model_selection import train_test_split

INPUT_ROOT = "../input/"
TRAIN_DIR = os.path.join(INPUT_ROOT, "train/")
image_size = 64
num_channels = 3
pixel_depth = 255
num_labels = 5
num_digits = 10
patch_size_3 = 3
depth = 32

digit_structure_path = os.path.join(TRAIN_DIR, "digitStruct.mat")

#Check if data exists
if not os.path.exists(digit_structure_path):
    data_loader.get_training_data(INPUT_ROOT, "train.tar.gz")

digit_struct = DigitStruct(digit_structure_path)
labels, paths = digit_struct.load_labels_and_paths()

labels = labels[:1000]
paths = paths[:1000]

image_paths = [TRAIN_DIR + s for s in paths]
images_normalized = image_helpers.prep_data(image_paths, image_size, num_channels, pixel_depth)

np.random.seed(42)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


def splitLengthsFromLabels(labels):
    lengths = np.array(labels[:,0].tolist())
    digit_labels = np.zeros((labels.shape[0], num_labels, num_digits))
    for i in range(0, len(labels[:, 1:])):
        current = labels[i, 1:]
        reshaped = np.array(current.tolist())
        digit_labels[i, :, :] = reshaped

    return digit_labels, lengths



train_dataset_rand, train_labels_rand = randomize(images_normalized, labels)
train_images, valid_images, train_labels, valid_labels = train_test_split(train_dataset_rand, train_labels_rand, train_size=0.9, random_state=0)

train_labels, train_lengths = splitLengthsFromLabels(train_labels) 
valid_labels, valid_lengths = splitLengthsFromLabels(valid_labels) 

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
        
        labels = tf.placeholder(tf.float32, shape=(None, num_labels, num_digits))
        lengths = tf.placeholder(tf.float32, shape=(None, 5))
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
        h_conv3 = conv2d_relu(h_conv3, w_conv4, b_conv4)
        h_pool4 = max_pool_2x2(h_conv3)

        #Conv->Relu->Conv->Relu->Conv->Relu->Pool
        w_conv5 = weight_layer("w_conv5", [patch_size_3, patch_size_3, depth * 4, depth * 4])
        b_conv5 = bias_variable("b_conv5", [depth * 4])
        h_conv5 = conv2d_relu(h_pool4, w_conv5, b_conv5)
        w_conv6 = weight_layer("w_conv6", [patch_size_3, patch_size_3, depth * 4, depth * 4])
        b_conv6 = bias_variable("b_conv6", [depth * 4])
        h_conv6 = conv2d_relu(h_conv5, w_conv6, b_conv6)
        w_conv7 = weight_layer("w_conv7", [patch_size_3, patch_size_3, depth * 4, depth * 8])
        b_conv7 = bias_variable("b_conv7", [depth * 8])
        h_pool7 = max_pool_2x2(h_conv6)

        #Dropout -> Fully Connected -> Dropout -> Fully Connected
        drop_1 = tf.nn.dropout(h_pool7, 1.0)
        shape = drop_1.get_shape().as_list()
        reshape = tf.reshape(drop_1, [-1, shape[1] * shape[2] * shape[3]])

        fc = 8192
        hl = 4096
        w_fc = weight_layer("w_fc", [fc, hl])
        b_fc = bias_variable("b_fc", [hl])
        h_fc = tf.matmul(reshape, w_fc) + b_fc

        w_l = weight_layer("w_l", [hl, 5])
        b_l = weight_layer("b_l", [5])
        z_l = tf.matmul(h_fc, w_l) + b_l

        w_s_1 = weight_layer("z_s_1", [hl, num_digits])
        b_s_1 = bias_variable("b_s_1", [num_digits])
        z_s_1 = tf.matmul(h_fc, w_s_1) + b_s_1
        
        w_s_2 = weight_layer("z_s_2", [hl, num_digits])
        b_s_2 = bias_variable("b_s_2", [num_digits])
        z_s_2 = tf.matmul(h_fc, w_s_2) + b_s_2
        
        w_s_3 = weight_layer("z_s_3", [hl, num_digits])
        b_s_3 = bias_variable("b_s_3", [num_digits])
        z_s_3 = tf.matmul(h_fc, w_s_3) + b_s_3

        w_s_4 = weight_layer("z_s_4", [hl, num_digits])
        b_s_4 = bias_variable("b_s_4", [num_digits])
        z_s_4 = tf.matmul(h_fc, w_s_4) + b_s_4

        w_s_5 = weight_layer("z_s_5", [hl, num_digits])
        b_s_5 = bias_variable("b_s_5", [num_digits])
        z_s_5 = tf.matmul(h_fc, w_s_5) + b_s_5

        labels1 = tf.squeeze(tf.slice(labels, [0, 0, 0], [-1, 1, num_digits]), axis=1)
        labels2 = tf.squeeze(tf.slice(labels, [0, 1, 0], [-1, 1, num_digits]), axis=1)
        labels3 = tf.squeeze(tf.slice(labels, [0, 2, 0], [-1, 1, num_digits]), axis=1)
        labels4 = tf.squeeze(tf.slice(labels, [0, 3, 0], [-1, 1, num_digits]), axis=1)
        labels5 = tf.squeeze(tf.slice(labels, [0, 4, 0], [-1, 1, num_digits]), axis=1)

        #We don't want to backpropagate cost on outputs that are 
        five_gate = tf.minimum(lengths[:,4], [1.0])
        four_gate = tf.maximum(lengths[:,3], five_gate)
        three_gate = tf.maximum(lengths[:,2], four_gate)
        two_gate = tf.maximum(lengths[:,1], three_gate)
        one_gate = tf.maximum(lengths[:,0], two_gate)

        cost_length = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lengths, logits=z_l))
        cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels1, logits=z_s_1)) * one_gate
        cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels2, logits=z_s_2)) * two_gate
        cost3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels3, logits=z_s_3)) * three_gate
        cost4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels4, logits=z_s_4)) * four_gate
        cost5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels5, logits=z_s_5)) * five_gate

        total_cost = cost_length + cost1 + cost2 + cost3 + cost4 + cost5

        length_prediction = tf.nn.softmax(z_l)
        train_prediction = tf.stack([
            tf.nn.softmax(z_s_1),
            tf.nn.softmax(z_s_2),
            tf.nn.softmax(z_s_3),
            tf.nn.softmax(z_s_4),
            tf.nn.softmax(z_s_5)
            ], axis=1)

        optimizer = tf.train.AdamOptimizer(0.00001).minimize(total_cost)

        with tf.Session(graph=graph) as session:
            num_steps = 10000
            batch_size = 32
            tf.global_variables_initializer().run()
            print("Initialized")

            for step in range(num_steps):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_images[offset:(offset + batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                batch_lengths = train_lengths[offset:(offset + batch_size), :]

                feed_dict = {input : batch_data, labels : batch_labels, lengths: batch_lengths}

                if step % 500 == 0:
                    _, l, predictions, length_preds = session.run([optimizer, total_cost, train_prediction, length_prediction], feed_dict=feed_dict)
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                    #Validation

                    v_steps = 5
                    v_batch_size = int(valid_images.shape[0] / v_steps)
                    v_preds = np.zeros_like(valid_labels)
                    for v_step in range(v_steps):
                        v_offset = (v_step * v_batch_size) 
                        v_batch_data = valid_images[v_offset:(v_offset + v_batch_size), :, :, :]
                        v_batch_labels = valid_labels[v_offset:(v_offset + v_batch_size),:]

                        feed_dict = {input : v_batch_data, labels : v_batch_labels}
                        _, l, predictions = session.run([optimizer, total_cost, train_prediction], feed_dict=feed_dict)
                        v_preds[v_offset: v_offset + predictions.shape[0],:,:] = predictions

                    #If we missed any validation images at the end, process them now
                    if v_steps * v_batch_size < valid_images.shape[0]:
                        v_offset = (v_steps * v_batch_size) 
                        v_batch_data = valid_images[v_offset:valid_images.shape[0] , :, :, :]
                        v_batch_labels = valid_labels[v_offset:valid_images.shape[0],:]

                        feed_dict = {input : v_batch_data, labels : v_batch_labels}
                        _, l, predictions = session.run([optimizer, total_cost, train_prediction], feed_dict=feed_dict)
                        v_preds[v_offset: v_offset + predictions.shape[0],:,:] = predictions

                    print('Valid accuracy: %.1f%%' % accuracy(v_preds, valid_labels))
                else:
                    _, l, predictions = session.run([optimizer, total_cost, train_prediction], feed_dict=feed_dict)


TrainConvNet()