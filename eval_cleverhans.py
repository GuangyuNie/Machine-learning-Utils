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
from cleverhans_model import MNISTmodel
from cleverhans.utils_tf import model_eval

# Set TF random seed to improve reproducibility
tf.set_random_seed(4557077)

# import attack method
from pgd_attack import LinfPGDAttack
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import ProjectedGradientDescent
# Determine attack method
attack_method = 'PGD'

# Set upd the data, hyperparameters, and the model
num_eval_examples = 10000
batch_size = 200
img_size = 28
model = MNISTmodel()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x_input = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, 10])
X_test = mnist.test.images.reshape(-1, img_size, img_size, 1)
Y_test = mnist.test.labels
# A function for evaluating a single checkpoint


def evaluate_checkpoint(filename):
  if attack_method == 'BIM':
        bim = BasicIterativeMethod(model)
        bim_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.,
                  'nb_iter': 50,
                  'eps_iter': .01}
        adv_x = bim.generate(x_image, **bim_params)
  elif attack_method == 'FGM':
          FGM_attack = FastGradientMethod(model)
          FGM_params = {'eps': 0.3,'clip_min': 0., 'clip_max':1.}
          adv_x = FGM_attack.generate(x_image,**FGM_params)
  elif attack_method == 'PGD':
        pgd = ProjectedGradientDescent(model)
        pgd_params = {'eps': 0.09, 'clip_min': 0., 'clip_max': 1.,
                  'nb_iter': 40,
                  'eps_iter': .01}
        adv_x = pgd.generate(x_image, **pgd_params)
  preds_adv = model.get_probs(adv_x)


  with tf.Session() as sess:
    # Restore the checkpoint
    saver = tf.train.Saver(var_list = model.all_variables)
    saver.restore(sess, filename)
    
    eval_par = {'batch_size': batch_size}
    t1 = time.time()
    acc = model_eval(sess, x_image, y, preds_adv, X_test, Y_test, args=eval_par)
    t2 = time.time()
    print("Took", t2 - t1, "seconds")
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)


if __name__ == "__main__":
    filename = './checkpoint/noatt_ckpt'
    evaluate_checkpoint(filename)
