"""
Module which implements decoding functions using multiple attentions
for RNN decoders.

See http://arxiv.org/abs/1606.07481
"""
# tests: lint

import tensorflow as tf
import numpy as np
from neuralmonkey.nn.projection import linear


class Attention(object):
    # pylint: disable=unused-argument,too-many-instance-attributes,too-many-arguments
    # For maintaining the same API as in CoverageAttention

    def __init__(self, attention_states, scope, dropout_placeholder,
                 input_weights=None, max_fertility=None):
        """Create the attention object.

        Args:
            attention_states: A Tensor of shape (batch x time x state_size)
                              with the output states of the encoder.
            scope: The name of the variable scope in the graph used by this
                   attention object.
            dropout_placeholder: A Tensor that contains the value of the dropout
                                 keep probability
            input_weights: (Optional) The padding weights on the input.
            max_fertility: (Optional) For the Coverage attention compatibilty,
                           maximum fertility of one word.
        """
        self.scope = scope
        self.attentions_in_time = []
        self.attention_states = tf.nn.dropout(attention_states,
                                              dropout_placeholder)
        self.input_weights = input_weights

        with tf.variable_scope(scope):
            self.attn_length = attention_states.get_shape()[1].value
            self.attn_size = attention_states.get_shape()[2].value

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape
            # before.
            self.att_states_reshaped = tf.reshape(
                self.attention_states,
                [-1, self.attn_length, 1, self.attn_size])

            # Size of query vectors for attention.
            self.attention_vec_size = self.attn_size

            # This variable corresponds to Bahdanau's U_a in the paper
            k = tf.get_variable(
                "AttnW", [1, 1, self.attn_size, self.attention_vec_size],
                initializer=tf.random_normal_initializer(stddev=0.001))

            self.hidden_features = tf.nn.conv2d(self.att_states_reshaped, k,
                                                [1, 1, 1, 1], "SAME")

            # pylint: disable=invalid-name
            # see comments on disabling invalid names below
            self.v = tf.get_variable(
                name="AttnV",
                shape=[self.attention_vec_size],
                initializer=tf.random_normal_initializer(stddev=.001))
            self.v_bias = tf.get_variable(
                "AttnV_b", [], initializer=tf.constant_initializer(0))

    def attention(self, query_state):
        """Put attention masks on att_states_reshaped
           using hidden_features and query.
        """

        with tf.variable_scope(self.scope + "/Attention") as varscope:
            # Sort-of a hack to get the matrix (bahdanau's W_a) in the linear
            # projection to be initialized this way. The biases are initialized
            # as zeros
            varscope.set_initializer(
                tf.random_normal_initializer(stddev=0.001))
            y = linear(query_state, self.attention_vec_size, scope=varscope)
            y = tf.reshape(y, [-1, 1, 1, self.attention_vec_size])

            # pylint: disable=invalid-name
            # code copied from tensorflow. Suggestion: rename the variables
            # according to the Bahdanau paper
            s = self.get_logits(y)

            if self.input_weights is None:
                a = tf.nn.softmax(s)
            else:
                a_all = tf.nn.softmax(s) * self.input_weights
                norm = tf.reduce_sum(a_all, 1, keep_dims=True) + 1e-8
                a = a_all / norm

            self.attentions_in_time.append(a)

            # Now calculate the attention-weighted vector d.
            d = tf.reduce_sum(tf.reshape(a, [-1, self.attn_length, 1, 1])
                              * self.att_states_reshaped, [1, 2])

            return tf.reshape(d, [-1, self.attn_size])

    def get_logits(self, y):
        # Attention mask is a softmax of v^T * tanh(...).
        return tf.reduce_sum(
            self.v * tf.tanh(self.hidden_features + y), [2, 3]) + self.v_bias

    # pylint: disable=no-self-use
    def feed_dict(self, dataset, train=False):
        return {}


class CoverageAttention(Attention):

    # pylint: disable=too-many-arguments
    # Great objects require great number of parameters
    def __init__(self, attention_states, scope, dropout_placeholder,
                 input_weights=None, max_fertility=5):

        super(CoverageAttention, self).__init__(attention_states, scope,
                                                dropout_placeholder,
                                                input_weights=input_weights,
                                                max_fertility=max_fertility)

        self.coverage_weights = tf.get_variable("coverage_matrix",
                                                [1, 1, 1, self.attn_size])
        self.fertility_weights = tf.get_variable("fertility_matrix",
                                                 [1, 1, self.attn_size])
        self.max_fertility = max_fertility

        self.fertility = 1e-8 + self.max_fertility * tf.sigmoid(
            tf.reduce_sum(self.fertility_weights * self.attention_states, [2]))

    def get_logits(self, y):
        coverage = sum(
            self.attentions_in_time) / self.fertility * self.input_weights

        logits = tf.reduce_sum(
            self.v * tf.tanh(
                self.hidden_features + y + self.coverage_weights * tf.reshape(
                    coverage, [-1, self.attn_length, 1, 1])),
            [2, 3])

        return logits


class GizaAttention(object):
    # pylint: disable=unused-argument

    def __init__(self, attention_states, scope, dropout_placeholder,
                 input_weights=None, max_fertility=None, data_id=None):
        self.scope = scope
        self.input_weights = input_weights
        self.data_id = data_id

        self.attentions_in_time = []

        with tf.variable_scope(scope):
            self.attn_length = attention_states.get_shape()[1].value
            self.attn_size = attention_states.get_shape()[2].value

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape
            # before.
            self.att_states_reshaped = tf.reshape(
                attention_states,
                [-1, self.attn_length, 1, self.attn_size])

    def attention(self, query_state):
        with tf.variable_scope(self.scope + "/Attention"):
            a = tf.placeholder(tf.float32, shape=[None, self.attn_length])
            self.attentions_in_time.append(a)

            # Now calculate the attention-weighted vector d.
            # pylint: disable=invalid-name
            d = tf.reduce_sum(tf.reshape(a, [-1, self.attn_length, 1, 1])
                              * self.att_states_reshaped, [1, 2])

            return tf.reshape(d, [-1, self.attn_size])

    def feed_dict(self, dataset, train=False):
        batch_size = len(dataset)
        attentions_in_time = [
            np.zeros([batch_size, self.attn_length], dtype=np.float32)
            for _ in self.attentions_in_time]

        with np.errstate(divide='ignore', invalid='ignore'):
            if train and dataset.has_series(self.data_id):
                for k, alignment in enumerate(dataset.get_series(self.data_id)):
                    attentions_in_time[0][k][0] = 1
                    for ali in alignment:
                        # This parsing should probably get done in Dataset
                        i, j = map(int, ali.split('-'))
                        if i < self.attn_length - 2 and \
                           j < len(attentions_in_time) - 1:
                            attentions_in_time[j+1][k][i+1] = 1

                for a in attentions_in_time:
                    a /= a.sum(axis=1, keepdims=True)
                    a[np.isnan(a)] = 0

        return dict(zip(self.attentions_in_time, attentions_in_time))
