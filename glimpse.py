from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import weight_variable, bias_variable


class GlimpseNet(object):
  """Glimpse network.

  Take glimpse location input and output features for RNN.

  """

  def __init__(self, config, images_ph):
    self.original_size = config.original_size
    self.num_channels = config.num_channels
    self.sensor_size = config.sensor_size
    self.win_size = config.win_size
    self.minRadius = config.minRadius
    self.depth = config.depth

    self.hg_size = config.hg_size
    self.hl_size = config.hl_size
    self.g_size = config.g_size
    self.loc_dim = config.loc_dim

    self.images_ph = images_ph

    self.init_weights()

  def init_weights(self):
    """ Initialize all the trainable weights."""
    self.w_g0 = weight_variable((self.sensor_size, self.hg_size))
    self.b_g0 = bias_variable((self.hg_size,))
    self.w_l0 = weight_variable((self.loc_dim, self.hl_size))
    self.b_l0 = bias_variable((self.hl_size,))
    self.w_g1 = weight_variable((self.hg_size, self.g_size))
    self.b_g1 = bias_variable((self.g_size,))
    self.w_l1 = weight_variable((self.hl_size, self.g_size))
    self.b_l1 = weight_variable((self.g_size,))

  def get_glimpse(self, loc):
    """Take glimpse on the original images."""
    imgs = tf.reshape(self.images_ph, [
        tf.shape(self.images_ph)[0], self.original_size, self.original_size,
        self.num_channels
    ])
    glimpse_imgs = tf.image.extract_glimpse(imgs,
                                            [self.win_size, self.win_size], loc)
    glimpse_imgs = tf.reshape(glimpse_imgs, [
        tf.shape(loc)[0], self.win_size * self.win_size * self.num_channels
    ])
    return glimpse_imgs

  '''
  #2 layer MLP
  def __call__(self, loc):
    glimpse_input = self.get_glimpse(loc)

    glimpse_input = tf.reshape(glimpse_input,
                               (tf.shape(loc)[0], self.sensor_size))
    g = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, self.w_g0, self.b_g0))
    g = tf.nn.xw_plus_b(g, self.w_g1, self.b_g1)
    l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
    l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)
    g = tf.nn.relu(g + l)
    return g
  '''

  #MNIST
  def deepnn(self, x):
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

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
      W_conv1 = self.weight_variable([5, 5, 3, 32])
      b_conv1 = self.bias_variable([32])
      h_conv1 = tf.nn.relu(self.conv2d(x, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
      h_pool1 = self.max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
      W_conv2 = self.weight_variable([5, 5, 32, 64])
      b_conv2 = self.bias_variable([64])
      h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
      h_pool2 = self.max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
      W_fc1 = self.weight_variable([7 * 7 * 64, 256])
      b_fc1 = self.bias_variable([256])

      h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
      h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

    return h_fc1_drop


  def conv2d(self, x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


  def max_pool_2x2(self, x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


  def weight_variable(self, shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


  def bias_variable(self, shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def __call__(self, loc):
    glimpse_input = self.get_glimpse(loc)

    glimpse_input = tf.reshape(glimpse_input, (tf.shape(loc)[0], self.win_size, self.win_size, self.num_channels))

    g = self.deepnn(glimpse_input)
    l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
    l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)
    g = tf.nn.relu(g + l)
    return g

class LocNet(object):
  """Location network.

  Take output from other network and produce and sample the next location.

  """

  def __init__(self, config):
    self.loc_dim = config.loc_dim
    self.input_dim = config.cell_output_size
    self.loc_std = config.loc_std
    self._sampling = True

    self.init_weights()

  def init_weights(self):
    self.w = weight_variable((self.input_dim, self.loc_dim))
    self.b = bias_variable((self.loc_dim,))

  def __call__(self, input):
    mean = tf.clip_by_value(tf.nn.xw_plus_b(input, self.w, self.b), -1., 1.)
    mean = tf.stop_gradient(mean)
    if self._sampling:
      loc = mean + tf.random_normal(
          (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
      loc = tf.clip_by_value(loc, -1., 1.)
    else:
      loc = mean
    loc = tf.stop_gradient(loc)
    return loc, mean

  @property
  def sampling(self):
    return self._sampling

  @sampling.setter
  def sampling(self, sampling):
    self._sampling = sampling
