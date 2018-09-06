
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os.path
import two_layer_fc
import numpy;
import data_helpers
import re
from PIL import Image
from tensorflow.python.tools import freeze_graph

PATH = ".\\banana_resize\\"
width = 0;
height = 0;
size = 0;
image = None;
image_data = None;
whole_data = None;
first = True;
file_list = os.listdir(PATH);

file_list.sort(key = lambda x : (x[0], int(re.split("(\\d*)", x)[3])));

for filename in file_list:
    image = Image.open(PATH + filename);
    image_data = numpy.array(image.getdata()).reshape(-1);

    if ( first == True ):
        whole_data = numpy.array( image_data );
        first = False;
    else:
        whole_data = numpy.vstack([whole_data, image_data]);

    image.close();
    print(filename);


# Model parameters as external flags
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for the training.')
flags.DEFINE_integer('max_steps', 500, 'Number of steps to run trainer.') # 2000
flags.DEFINE_integer('hidden1', 120, 'Number of units in hidden layer 1.') # 120
flags.DEFINE_integer('hidden2', 20, 'Number of units in hidden layer 2.') # 120
flags.DEFINE_integer('batch_size', 128,
  'Batch size. Must divide dataset sizes without remainder.')
flags.DEFINE_string('train_dir', 'tf_logs',
  'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')

FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
  print('{} = {}'.format(attr, value))
print()

beginTime = time.time()

# Put logs for each run in separate directory
logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

# Uncommenting these lines removes randomness
# You'll get exactly the same result on each run
# np.random.seed(1)
# tf.set_random_seed(1)

TRAIN_PERCENT = 0.90;

image_data = whole_data;
image_shape = image_data.shape;
image_rows = image_shape[0];
image_train_rows = int(image_rows * TRAIN_PERCENT);

IMAGE_PIXELS = image_shape[1];
CLASSES = 6

data_sets = {};
data_sets['images_train'] = image_data[0 : image_train_rows, :];
data_sets['images_test'] = image_data[image_train_rows : image_rows, :];

label_data = numpy.loadtxt(PATH + "label.txt");
data_sets['labels_train'] = label_data[0 : image_train_rows];
data_sets['labels_test'] = label_data[image_train_rows : image_rows];


# Create the model
x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, CLASSES])

# Build the graph for the deep net
y_conv, keep_prob = two_layer_fc.deepnn(x)

with tf.name_scope('loss'):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                          logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

graph_location = logdir
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
  batches = data_helpers.gen_batch(list(zipped_data), FLAGS.batch_size, FLAGS.max_steps)

  for i in range(FLAGS.max_steps):
    batch = next(batches)

    images_batch, labels_batch = zip(*batch)

    # reshape
    labels_batch_reshape = np.zeros((len(labels_batch), CLASSES))
    for labels_index in range(0, len(labels_batch)):
      labels_batch_reshape[labels_index][int(labels_batch[labels_index])] = 1

    #print ( images_batch )
    #print ( labels_batch_reshape )

    if i % 1 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: images_batch, y_: labels_batch_reshape, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
      
      #result = y_conv.eval( feed_dict={ x: images_batch, y_: labels_batch_reshape, keep_prob: 1.0} )
      #print( result ) 
    train_step.run(feed_dict={x: images_batch, y_: labels_batch_reshape, keep_prob: 0.5})


  labels_batch_reshape = np.zeros((len(data_sets['labels_test']), CLASSES))
  for labels_index in range(0, len(data_sets['labels_test'])):
    labels_batch_reshape[labels_index][int(data_sets['labels_test'][labels_index])] = 1

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: data_sets['images_test'], y_: labels_batch_reshape, keep_prob: 1.0}))

  result = y_conv.eval( feed_dict={ x: data_sets['images_test'], y_: labels_batch_reshape, keep_prob: 1.0} )
  print( result ) 

  saver = tf.train.Saver()
  tf.train.write_graph(sess.graph_def, '.', 'tfdroid.pbtxt')  
  saver.save(sess, '.\\tfdroid.ckpt')

  exit()

# -----------------------------------------------------------------------------
# Prepare the Tensorflow graph
# (We're only defining the graph here, no actual calculations taking place)
# -----------------------------------------------------------------------------

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS],
  name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')

# Operation for the classifier's result
logits = two_layer_fc.inference(images_placeholder, IMAGE_PIXELS,
  FLAGS.hidden1, FLAGS.hidden2, CLASSES, reg_constant=FLAGS.reg_constant)

# Operation for the loss function
loss = two_layer_fc.loss(logits, labels_placeholder)

# Operation for the training step
train_step = two_layer_fc.training(loss, FLAGS.learning_rate)

# Operation calculating the accuracy of our predictions
accuracy = two_layer_fc.evaluation(logits, labels_placeholder)

# Operation merging summary data for TensorBoard
summary = tf.summary.merge_all()

# Define saver to save model state at checkpoints
saver = tf.train.Saver()

# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------

with tf.Session() as sess:
  # Initialize variables and create summary-writer
  sess.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter(logdir, sess.graph)

  # Generate input data batches
  zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
  batches = data_helpers.gen_batch(list(zipped_data), FLAGS.batch_size,
    FLAGS.max_steps)

  for i in range(FLAGS.max_steps):

    # Get next input data batch
    batch = next(batches)
    images_batch, labels_batch = zip(*batch)
    feed_dict = {
      images_placeholder: images_batch,
      labels_placeholder: labels_batch
    }

    # Periodically print out the model's current accuracy
    if i % 1 == 0:
      train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
      print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
      summary_str = sess.run(summary, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)

    # Perform a single training step
    sess.run([train_step, loss], feed_dict=feed_dict)

    # Periodically save checkpoint
    #if (i + 1) % 10 == 0:
    #  checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
    #  saver.save(sess, checkpoint_file, global_step=i)
    #  print('Saved checkpoint')

  # After finishing the training, evaluate on the test set
  test_accuracy = sess.run(accuracy, feed_dict={
    images_placeholder: data_sets['images_test'],
    labels_placeholder: data_sets['labels_test']})
  print('Test accuracy {:g}'.format(test_accuracy))

  print(sess.run(logits, feed_dict={
    images_placeholder: data_sets['images_test'],
    labels_placeholder: data_sets['labels_test']}))


  train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train', sess.graph)
  tf.train.write_graph(sess.graph_def, '.', 'tfdroid.pbtxt')  
  saver.save(sess, '.\\tfdroid.ckpt')

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))

MODEL_NAME = 'tfdroid'

# Freeze the graph

input_graph_path = MODEL_NAME+'.pbtxt'
checkpoint_path = './'+MODEL_NAME+'.ckpt'
input_saver_def_path = ""
input_binary = False
output_node_names = "accuracy/Cast"#"O"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")


























#width = 0;
#height = 0;
#size = 0;
#image = None;
#image_data = None;
#whole_data = None;
#first = True;
#file_list = os.listdir(PATH);

#file_list.sort(key = lambda x : int(re.split("(\\d*)", x)[3]));

#for filename in file_list:
#    image = Image.open(PATH + filename);
#    image_data = numpy.array(image.getdata()).reshape(-1);

#    if ( first == True ):
#        whole_data = numpy.array( image_data );
#        first = False;
#    else:
#        whole_data = numpy.vstack([whole_data, image_data]);

#    image.close();
#    print(filename);


#numpy.save("data", whole_data);


#PATH = ".\\banana_resize\\"
#size = 224, 224

#width = 0;
#height = 0;
#image = None;
#image_data = None;
#whole_data = None;
#first = True;
#file_list = os.listdir(PATH);

#file_list.sort(key = lambda x : int(re.split("(\\d*)", x)[3]));

#for filename in file_list:
#    image = Image.open(PATH + filename);
#    image = image.rotate( 90 );
#    image.save( PATH + "A_" + filename , "JPEG" );

#    image = image.rotate( 90 );
#    image.save( PATH + "B_" + filename , "JPEG" );

#    image = image.rotate( 90 );
#    image.save( PATH + "C_" + filename , "JPEG" );

#    image.close();
#    print(filename);


#PATH = ".\\banana_resize\\"
#size = 224, 224

#width = 0;
#height = 0;
#image = None;
#image_data = None;
#whole_data = None;
#first = True;
#file_list = os.listdir(PATH);

#file_list.sort(key = lambda x : int(re.split("(\\d*)", x)[3]));

#for filename in file_list:
#    image = Image.open(PATH + filename);
#    image = image.resize( size, Image.ANTIALIAS );
#    image.save( PATH + filename, "JPEG" );

#    image.close();
#    print(filename);










