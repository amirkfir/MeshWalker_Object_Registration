from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict
import glob

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.legacy_tf_layers.core import dropout

import utils

from tensorflow.keras.initializers import Orthogonal, GlorotNormal, HeNormal,GlorotUniform

from tensorflow import keras
from attention_layer import Attention
layers = tf.keras.layers




class RnnWalkBase(tf.keras.Model):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn=None,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    super(RnnWalkBase, self).__init__(name='')

    self._classes = classes
    self._params = params
    self._model_must_be_load = model_must_be_load
    self._init_layers()
    inputs = tf.keras.layers.Input(shape=(2,100, net_input_dim))
    self.build(input_shape=(1,2, 100, net_input_dim))
    outputs = self.call(inputs)
    if dump_model_visualization:
      tmp_model = keras.Model(inputs=inputs, outputs=outputs, name='WalkModel')
      tmp_model.summary(print_fn=self._print_fn)
      tf.keras.utils.plot_model(tmp_model, params.logdir + '/RnnWalkModel.png', show_shapes=True)

    self.manager = None
    if optimizer:
      if model_fn:
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
      else:
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
      self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self._params.logdir, max_to_keep=5)
      if model_fn: # Transfer learning
        self.load_weights(model_fn)
        self.checkpoint.optimizer = optimizer
      else:
        self.load_weights()
    else:
      self.checkpoint = tf.train.Checkpoint(model=self)
      if model_fn is None:
        model_fn = self._get_latest_keras_model()
      self.load_weights(model_fn)

  def _print_fn(self, st):
    with open(self._params.logdir + '/log.txt', 'at') as f:
      f.write(st + '\n')

  def _get_latest_keras_model(self):
    filenames = glob.glob(self._params.logdir + '/*model2keep__*')
    # iters_saved = [int(f.split('model2keep__')[-1].split('.keras')[0]) for f in filenames]
    iters_saved = [int(f.split('model2keep__')[-1][-14:].split('.keras')[0]) for f in filenames]
    return filenames[np.argmax(iters_saved)]

  def load_weights(self, filepath=None):
    if filepath is not None and filepath.endswith('.keras'):
      super(RnnWalkBase, self).load_weights(filepath)
    elif filepath is None:
      _ = self.checkpoint.restore(self.manager.latest_checkpoint)
      print(utils.color.BLUE, 'Starting from iteration: ', self.checkpoint.optimizer.iterations.numpy(), utils.color.END)
    else:
      filepath = filepath.replace('//', '/')
      _ = self.checkpoint.restore(filepath)

  def save_weights(self, folder, step=None, keep=False, name = ""):
    if self.manager is not None:
      self.manager.save()
    if keep:
      super(RnnWalkBase, self).save_weights(folder + '/learned_model2keep__'+ name + str(step).zfill(8) + '.keras')


class RnnWalkNet_single_line(RnnWalkBase):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    if params.layer_sizes is None:
      self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 512, 'gru2': 1024, 'gru3': 1024}
    else:
      self._layer_sizes = params.layer_sizes
    super(RnnWalkNet_single_line, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization, optimizer=optimizer)

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0)#(0.0001)


    self._use_norm_layer = self._params.use_norm_layer is not None
    if self._params.use_norm_layer == 'InstanceNorm':
      self._norm1 = tfa.layers.InstanceNormalization(axis=2)
      self._norm2 = tfa.layers.InstanceNormalization(axis=2)
    elif self._params.use_norm_layer == 'BatchNorm':
      self._norm1 = layers.BatchNormalization(axis=2)
      self._norm2 = layers.BatchNormalization(axis=2)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=HeNormal())
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=HeNormal())
    # rnn_layer = layers.LSTM #layers.GRU
    # self._gru1 = rnn_layer(self._layer_sizes['gru1'], return_sequences=True, return_state=False,
    #                         dropout=self._params.net_gru_dropout,
    #                         recurrent_initializer=Orthogonal(), kernel_initializer=GlorotUniform(),
    #                         kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    self._gru1_norm = layers.LayerNormalization()
    # self._gru2 = rnn_layer(self._layer_sizes['gru2'], return_sequences=True, return_state=False,
    #                         dropout=self._params.net_gru_dropout,
    #                         recurrent_initializer=Orthogonal(), kernel_initializer=GlorotUniform(),
    #                         kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    self._gru2_norm = layers.LayerNormalization()
    #
    # self._gru3 = rnn_layer(self._layer_sizes['gru3'],
    #                        return_sequences=False,
    #                        return_state=False,
    #                        dropout=self._params.net_gru_dropout,
    #                        recurrent_initializer=Orthogonal(), kernel_initializer=GlorotUniform(),
    #                        kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
    #                        bias_regularizer=kernel_regularizer)
    # second walk lane:
    self._gru3_norm = layers.LayerNormalization()
    self._attention0 = Attention()
    self._attention1 = Attention()
    self._attention2 = Attention()

    if self._params.use_norm_layer == 'InstanceNorm':
      self._2_norm1 = tfa.layers.InstanceNormalization(axis=2)
      self._2_norm2 = tfa.layers.InstanceNormalization(axis=2)
    elif self._params.use_norm_layer == 'BatchNorm':
      self._2_norm1 = layers.BatchNormalization(axis=2)
      self._2_norm2 = layers.BatchNormalization(axis=2)
    self._2_fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=HeNormal())
    self._2_fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=HeNormal())
    # self._2_gru1 = rnn_layer(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
    #                         dropout=self._params.net_gru_dropout,
    #                         recurrent_initializer=Orthogonal(), kernel_initializer=GlorotUniform(),
    #                         kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    # self._2_gru1_norm = layers.LayerNormalization()
    # self._2_gru2 = rnn_layer(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
    #                         dropout=self._params.net_gru_dropout,
    #                         recurrent_initializer=Orthogonal(), kernel_initializer=GlorotUniform(),
    #                         kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    # self._2_gru2_norm = layers.LayerNormalization()
    # self._2_gru3 = rnn_layer(self._layer_sizes['gru3'], time_major=False,
    #                        return_sequences=not self._params.one_label_per_model,
    #                        return_state=False,
    #                        dropout=self._params.net_gru_dropout,
    #                        recurrent_initializer=Orthogonal(), kernel_initializer=GlorotUniform(),
    #                        kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
    #                        bias_regularizer=kernel_regularizer)
    # self._2_gru3_norm = layers.LayerNormalization()

    self._concat_norm = layers.LayerNormalization()
    self._fc_0 = layers.Dense(512, activation=None, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                     kernel_initializer=HeNormal())
    self._fc_0_norm = layers.LayerNormalization()
    self._fc_1 = layers.Dense(256, activation=None, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                     kernel_initializer=HeNormal())
    self._fc_1_norm = layers.LayerNormalization()

    self._fc_2 = layers.Dense(256, activation=None, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                     kernel_initializer=HeNormal())
    self._fc_2_norm = layers.LayerNormalization()

    self._fc_3 = layers.Dense(12, activation=None, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=HeNormal())
    self._pooling = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

  def call(self, model_ftrs, classify=True, skip_1st=True, training=True):
    if skip_1st:
        x = model_ftrs[:, 0, 1:]
        y = model_ftrs[:, 1, 1:]
    else:
        x = model_ftrs[:, 0, :]
        y = model_ftrs[:, 1, :]

    # x = tf.concat([x, y], axis=-1)
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.leaky_relu(x)
    # x = tf.nn.relu(x)
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.leaky_relu(x)
    # x = tf.nn.relu(x)
    y = self._2_fc1(y)
    if self._use_norm_layer:
      y = self._2_norm1(y, training=training)
    y = tf.nn.leaky_relu(y)
    # x = tf.nn.relu(x)
    y = self._2_fc2(y)
    if self._use_norm_layer:
      y = self._2_norm2(y, training=training)
    y = tf.nn.leaky_relu(y)


    x1 = self._attention0([x,y], training=training)
    x1 = self._gru1_norm(x1)
    x2 = self._attention1([x1, x1], training=training)
    x2 = self._gru2_norm(x2)
    x3 = self._attention2([x2,x2], training=training)
    x3 = self._gru3_norm(x3)
    x = x3


    z = self._fc_0(x)
    # z = self._fc_0_norm(z)
    # z = tf.nn.leaky_relu(z)
    z = tf.nn.relu(z)
    # z = tf.nn.dropout(z, rate=0.05)
    z = self._fc_1(z)
    # z = self._fc_1_norm(z)
    # z = tf.nn.leaky_relu(z)
    z = tf.nn.relu(z)
    # z = tf.nn.dropout(z, rate=0.05)
    z = self._fc_2(z)
    # z = self._fc_2_norm(z)
    # z = tf.nn.leaky_relu(z)
    z = tf.nn.relu(z)
    # z = tf.nn.dropout(z, rate=0.05)
    z = self._fc_3(z)
    return z

class RnnWalkNet(RnnWalkBase):
    def __init__(self,
                 params,
                 classes,
                 net_input_dim,
                 model_fn,
                 model_must_be_load=False,
                 dump_model_visualization=True,
                 optimizer=None):
      if params.layer_sizes is None:
        self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 512, 'gru2': 1024, 'gru3': 1024}
      else:
        self._layer_sizes = params.layer_sizes
      super(RnnWalkNet, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load,
                                       dump_model_visualization=dump_model_visualization, optimizer=optimizer)

    def _init_layers(self):
      kernel_regularizer = tf.keras.regularizers.l2(0)  # (0.0001)

      self._use_norm_layer = self._params.use_norm_layer is not None
      if self._params.use_norm_layer == 'InstanceNorm':
        self._norm1 = tfa.layers.InstanceNormalization(axis=2)
        self._norm2 = tfa.layers.InstanceNormalization(axis=2)
      elif self._params.use_norm_layer == 'BatchNorm':
        self._norm1 = layers.BatchNormalization(axis=2)
        self._norm2 = layers.BatchNormalization(axis=2)
      self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer,
                               bias_regularizer=kernel_regularizer,
                               kernel_initializer=HeNormal())
      self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer,
                               bias_regularizer=kernel_regularizer,
                               kernel_initializer=HeNormal())
      rnn_layer = layers.LSTM  # layers.GRU
      self._gru1 = rnn_layer(self._layer_sizes['gru1'], return_sequences=True, return_state=False,
                             dropout=self._params.net_gru_dropout,
                             recurrent_initializer=Orthogonal(), kernel_initializer=GlorotUniform(),
                             kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                             bias_regularizer=kernel_regularizer)
      self._gru1_norm = layers.LayerNormalization()
      self._gru2 = rnn_layer(self._layer_sizes['gru2'], return_sequences=True, return_state=False,
                             dropout=self._params.net_gru_dropout,
                             recurrent_initializer=Orthogonal(), kernel_initializer=GlorotUniform(),
                             kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                             bias_regularizer=kernel_regularizer)
      self._gru2_norm = layers.LayerNormalization()

      self._gru3 = rnn_layer(self._layer_sizes['gru3'],
                             return_sequences=False,
                             return_state=False,
                             dropout=self._params.net_gru_dropout,
                             recurrent_initializer=Orthogonal(), kernel_initializer=GlorotUniform(),
                             kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                             bias_regularizer=kernel_regularizer)
      # second walk lane:
      self._gru3_norm = layers.LayerNormalization()
      if self._params.use_norm_layer == 'InstanceNorm':
        self._2_norm1 = tfa.layers.InstanceNormalization(axis=2)
        self._2_norm2 = tfa.layers.InstanceNormalization(axis=2)
      elif self._params.use_norm_layer == 'BatchNorm':
        self._2_norm1 = layers.BatchNormalization(axis=2)
        self._2_norm2 = layers.BatchNormalization(axis=2)
      self._2_fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=kernel_regularizer,
                                 kernel_initializer=HeNormal())
      self._2_fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=kernel_regularizer,
                                 kernel_initializer=HeNormal())
      self._2_gru1 = rnn_layer(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                               dropout=self._params.net_gru_dropout,
                               recurrent_initializer=Orthogonal(), kernel_initializer=GlorotUniform(),
                               kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                               bias_regularizer=kernel_regularizer)
      self._2_gru1_norm = layers.LayerNormalization()
      self._2_gru2 = rnn_layer(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                               dropout=self._params.net_gru_dropout,
                               recurrent_initializer=Orthogonal(), kernel_initializer=GlorotUniform(),
                               kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                               bias_regularizer=kernel_regularizer)
      self._2_gru2_norm = layers.LayerNormalization()
      self._2_gru3 = rnn_layer(self._layer_sizes['gru3'], time_major=False,
                               return_sequences=not self._params.one_label_per_model,
                               return_state=False,
                               dropout=self._params.net_gru_dropout,
                               recurrent_initializer=Orthogonal(), kernel_initializer=GlorotUniform(),
                               kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                               bias_regularizer=kernel_regularizer)
      self._2_gru3_norm = layers.LayerNormalization()

      self._concat_norm = layers.LayerNormalization()
      self._fc_0 = layers.Dense(512, activation=None, kernel_regularizer=kernel_regularizer,
                                bias_regularizer=kernel_regularizer,
                                kernel_initializer=HeNormal())
      self._fc_0_norm = layers.LayerNormalization()
      self._fc_1 = layers.Dense(256, activation=None, kernel_regularizer=kernel_regularizer,
                                bias_regularizer=kernel_regularizer,
                                kernel_initializer=HeNormal())
      self._fc_1_norm = layers.LayerNormalization()

      self._fc_2 = layers.Dense(256, activation=None, kernel_regularizer=kernel_regularizer,
                                bias_regularizer=kernel_regularizer,
                                kernel_initializer=HeNormal())
      self._fc_2_norm = layers.LayerNormalization()

      self._fc_3 = layers.Dense(12, activation=None, kernel_regularizer=kernel_regularizer,
                                bias_regularizer=kernel_regularizer,
                                kernel_initializer=HeNormal())
      self._pooling = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

    def call(self, model_ftrs, classify=True, skip_1st=True, training=True):
      if skip_1st:
        x = model_ftrs[:, 0, 1:]
        y = model_ftrs[:, 1, 1:]
      else:
        x = model_ftrs[:, 0]
        y = model_ftrs[:, 1]
      x = self._fc1(x)
      y = self._2_fc1(y)
      if self._use_norm_layer:
        x = self._norm1(x, training=training)
        y = self._2_norm1(y, training=training)
      x = tf.nn.leaky_relu(x)
      y = tf.nn.leaky_relu(y)
      x = self._fc2(x)
      y = self._2_fc2(y)
      if self._use_norm_layer:
        x = self._norm2(x, training=training)
        y = self._2_norm2(y, training=training)
      x = tf.nn.leaky_relu(x)
      y = tf.nn.leaky_relu(y)

      x1 = self._gru1(x, training=training)
      x1 = self._gru1_norm(x1)
      x2 = self._gru2(x1, training=training)
      x2 = self._gru2_norm(x2)
      x3 = self._gru3(x2, training=training)
      x3 = self._gru3_norm(x3)
      x = x3

      y1 = self._2_gru1(y, training=training)
      y1 = self._2_gru1_norm(y1)
      y2 = self._2_gru2(y1, training=training)
      y2 = self._2_gru2_norm(y2)
      y3 = self._2_gru3(y2, training=training)
      y3 = self._2_gru3_norm(y3)
      y = y3

      z = tf.concat([x, y], axis=-1)
      # z = self._concat_norm(z)
      z = self._fc_0(z)
      # z = self._fc_0_norm(z)
      z = tf.nn.leaky_relu(z)
      # z = tf.nn.dropout(z, rate=0.05)
      z = self._fc_1(z)
      # z = self._fc_1_norm(z)
      z = tf.nn.leaky_relu(z)
      # z = tf.nn.dropout(z, rate=0.05)
      z = self._fc_2(z)
      # z = self._fc_2_norm(z)
      z = tf.nn.leaky_relu(z)
      # z = tf.nn.dropout(z, rate=0.05)
      z = self._fc_3(z)
      return z

