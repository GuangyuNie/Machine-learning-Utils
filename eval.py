"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from model_mnist import Model_madry, Model_att, Model_crop
from pgd_attack import LinfPGDAttack

# Set upd the data, hyperparameters, and the model
num_eval_examples = 10000
batch_size = 16
img_size = 28
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
input_images = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size, 1))
input_label = tf.placeholder(tf.int64, shape=(batch_size))
model = Model_crop(input_images, input_label)
attack = LinfPGDAttack(model,
                       epsilon = 0.3,
                       k = 40,
                       a = 0.01,
                       random_start=True,
                       loss_func = 'xent')
# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  with tf.Session() as sess:
    # Restore the checkpoint
    saver = tf.train.Saver()
    saver.restore(sess, filename)

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / batch_size))

    test_adv_acc = []
    test_adv_loss = []
    test_nat_acc = []
    test_nat_loss = []
    x_adv = []
    y_adv = []
    for ibatch in range(num_batches):
      bstart = ibatch * batch_size
      bend = min(bstart + batch_size, num_eval_examples)
      x_batch_test = mnist.test.images[bstart:bend, :]
      y_batch_test = mnist.test.labels[bstart:bend]
      # clean testing image
      nat_dict_test = {input_images: x_batch_test.reshape(batch_size, img_size, img_size, 1),
                       input_label: y_batch_test}
      y_pred, test_nat_loss_i = sess.run([model.y_pred, model.xent], feed_dict=nat_dict_test)
      counts = np.asarray([np.argmax(np.bincount(y_pred[:, i])) for i in range(batch_size)])
      test_nat_acc_i = np.mean(counts == nat_dict_test[input_label])
      test_nat_acc += [test_nat_acc_i]
      test_nat_loss += [test_nat_loss_i]
      # adversarial testing image
      x_batch_test_adv = attack.perturb(x_batch_test.reshape(batch_size, img_size, img_size, 1), y_batch_test, sess)
      adv_dict_test = {input_images: x_batch_test_adv.reshape(batch_size, img_size, img_size, 1),
                       input_label: y_batch_test}

      x_adv += [x_batch_test_adv]
      y_adv += [y_batch_test]      
      y_pred, test_adv_loss_i = sess.run([model.y_pred, model.xent], feed_dict=adv_dict_test)
      counts = np.asarray([np.argmax(np.bincount(y_pred[:, i])) for i in range(batch_size)])
      test_adv_acc_i = np.mean(counts == adv_dict_test[input_label])
      test_adv_acc += [test_adv_acc_i]
      test_adv_loss += [test_adv_loss_i]
    file_id = os.path.join('./adv_mnist.npy')
    x_adv = np.concatenate(x_adv,0)
    y_adv = np.concatenate(y_adv,0)
    save_i = {'image': x_adv,'label': y_adv}
    test_adv_acc = 100 * np.mean(test_adv_acc)
    print('natural: {:.2f}%'.format(100 * np.mean(test_nat_acc)))
    print('adversarial: {:.2f}%'.format(test_adv_acc))
    print('avg nat loss: {:.4f}'.format(np.mean(test_nat_loss)))
    print('avg adv loss: {:.4f}'.format(np.mean(test_adv_loss)))


if __name__ == "__main__":
    filename = './checkpoint/noatt_ckpt'
    evaluate_checkpoint(filename)
