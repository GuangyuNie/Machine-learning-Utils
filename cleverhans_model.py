from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import tensorflow as tf
from cleverhans.model import Model
from cleverhans.utils import deterministic_dict
from cleverhans.dataset import Factory, MNIST
import numpy as np
class MNISTmodel(Model):

    def __init__(self, nb_classes=10):
        # we cannot use scopes, give these variables names, etc.
        self.W_conv1 = self._weight_variable([5, 5, 1, 32],name = 'classifier/conv1/weights')
        self.b_conv1 = self._bias_variable([32],name = 'classifier/conv1/biases')
        self.W_conv2 = self._weight_variable([5, 5, 32, 64],name = 'classifier/conv2/weights')
        self.b_conv2 = self._bias_variable([64],name = 'classifier/conv2/biases')
        self.W_fc1 = self._weight_variable([1600, 1024],name = 'classifier/fc1/weights')
        self.b_fc1 = self._bias_variable([1024],name = 'classifier/fc1/biases')
        self.W_fc2 = self._weight_variable([1024, nb_classes],name = 'classifier/fc2/weights')
        self.b_fc2 = self._bias_variable([nb_classes],name = 'classifier/fc2/biases')
        self.all_variables = [self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2]
        #self.xent = tf.Variable(tf.zeros([16,10]))
        Model.__init__(self, '', nb_classes, {})
        #self.dataset_factory = Factory(MNIST, {"center": False})
    def get_params(self):
        return [
            self.W_conv1,
            self.b_conv1,
            self.W_conv2,
            self.b_conv2,
            self.W_fc1,
            self.b_fc1,
            self.W_fc2,
            self.b_fc2]


    def fprop(self, x):
        self.x_input = x
        #self.y_input = y
        self.x_voting = []
        self.x_crop = []
        self.xent_=[]
        self.probs_ = []
        self.loc = np.arange(10,18,1, dtype='int64')
        self.loc = [(i, j) for i in self.loc for j in self.loc]
        #if idx!=None:
         #   self.loc = self.loc[idx:idx+1]
        output = OrderedDict()
        for i, loc_i in enumerate(self.loc):
            loc_x, loc_y = loc_i
            x_crop_i = self.x_input[:, loc_x-10:loc_x+10, loc_y-10:loc_y+10, :]
            self.x_crop += [x_crop_i]
            # first convolutional layer 
            h_conv1 = tf.nn.relu(self._conv2d(x_crop_i, self.W_conv1) + self.b_conv1)
            h_pool1 = self._max_pool_2x2(h_conv1)
            # second convolutional layer
            h_conv2 = tf.nn.relu(self._conv2d(h_pool1, self.W_conv2) + self.b_conv2)
            h_pool2 = self._max_pool_2x2(h_conv2)

            # first fully connected layer
            h_pool2_flat = tf.reshape(h_pool2, [-1, 1600])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

            # output layer
            logits = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
            probs = tf.nn.softmax(logits=logits)
            self.xent_ += [logits]
            self.probs_ += [probs]
            tf.get_variable_scope().reuse_variables()
            assert tf.get_variable_scope().reuse == True
        self.xent_ = tf.reduce_mean(self.xent_,0)
        self.probs_ = tf.reduce_mean(self.probs_,0)
        output = deterministic_dict(locals())
        del output["self"]
        output[self.O_LOGITS] = self.xent_
        output[self.O_PROBS] = self.probs_
        return output


    @staticmethod
    def _weight_variable(shape,name = None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial,name = name)

    @staticmethod
    def _bias_variable(shape,name = None):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial,name = name)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')

