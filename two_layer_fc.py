'''Builds a 2-layer fully-connected neural network'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

CLASSES = 6

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 224, 224, 3])
    tf.Print( x_image, [x_image] )

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 512 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([14 * 14 * 64, 512])
    b_fc1 = bias_variable([512])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 14*14*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
     keep_prob = tf.placeholder(tf.float32)
     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 512 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([512, CLASSES])
    b_fc2 = bias_variable([CLASSES])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)







def inference(images, image_pixels, hidden_units1, hidden_units2, classes, reg_constant=0):
  '''Build the model up to where it may be used for inference.

  Args:
      images: Images placeholder (input data).
      image_pixels: Number of pixels per image.
      hidden_units: Size of the first (hidden) layer.
      classes: Number of possible image classes/labels.
      reg_constant: Regularization constant (default 0).

  Returns:
      logits: Output tensor containing the computed logits.
  '''

  # Layer 1
  with tf.variable_scope('Layer1'):
    # Define the variables
    weights = tf.get_variable(
      name='weights',
      shape=[image_pixels, hidden_units1],
      initializer=tf.truncated_normal_initializer(
        stddev=1.0 / np.sqrt(float(image_pixels))),
      regularizer=tf.contrib.layers.l2_regularizer(reg_constant)
    )

    biases = tf.Variable(tf.zeros([hidden_units1]), name='biases')

    # Define the layer's output
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

  # Layer 2
  with tf.variable_scope('Layer2'):
    # Define the variables
    weights = tf.get_variable(
      name='weights',
      shape=[hidden_units1, hidden_units2],
      initializer=tf.truncated_normal_initializer(
        stddev=1.0 / np.sqrt(float(hidden_units1))),
      regularizer=tf.contrib.layers.l2_regularizer(reg_constant)
    )

    biases = tf.Variable(tf.zeros([hidden_units2]), name='biases')

    # Define the layer's output
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)


  # Layer 3
  with tf.variable_scope('Layer3'):
    # Define variables
    weights = tf.get_variable('weights', [hidden_units2, classes],
      initializer=tf.truncated_normal_initializer(
        stddev=1.0 / np.sqrt(float(hidden_units2))),
      regularizer=tf.contrib.layers.l2_regularizer(reg_constant))

    biases = tf.Variable(tf.zeros([classes]), name='biases')

    # Define the layer's output
    logits = tf.matmul(hidden2, weights) + biases

    # Define summery-operation for 'logits'-variable
    tf.summary.histogram('logits', logits)

  return logits


def loss(logits, labels):
  '''Calculates the loss from logits and labels.

  Args:
    logits: Logits tensor, float - [batch size, number of classes].
    labels: Labels tensor, int64 - [batch size].

  Returns:
    loss: Loss tensor of type float.
  '''

  with tf.name_scope('Loss'):
    # Operation to determine the cross entropy between logits and labels
    cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy'))

    # Operation for the loss function
    loss = cross_entropy + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # Add a scalar summary for the loss
    tf.summary.scalar('loss', loss)

  return loss


def training(loss, learning_rate):
  '''Sets up the training operation.

  Creates an optimizer and applies the gradients to all trainable variables.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_step: The op for training.
  '''

  # Create a variable to track the global step
  global_step = tf.Variable(0, name='global_step', trainable=False)

  # Create a gradient descent optimizer
  # (which also increments the global step counter)
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    loss, global_step=global_step)

  return train_step


def evaluation(logits, labels):
  '''Evaluates the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch size, number of classes].
    labels: Labels tensor, int64 - [batch size].

  Returns:
    accuracy: the percentage of images where the class was correctly predicted.
  '''

  with tf.name_scope('Accuracy'):
    # Operation comparing prediction with true label
    correct_prediction = tf.equal(tf.argmax(logits,1), labels)

    # Operation calculating the accuracy of the predictions
    accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Summary operation for the accuracy
    tf.summary.scalar('train_accuracy', accuracy)

  return accuracy
