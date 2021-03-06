from typing import cast, Any, Callable, Iterable, Optional, List

import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.nn.mlp import MultilayerPerceptron

# tests: lint, mypy

# pylint: disable=too-many-instance-attributes


class SequenceClassifier(ModelPart):
    """A simple MLP classifier over encoders.

    The API pretends it is an RNN decoder which always generates a sequence of
    length exactly one.
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoders: List[Any],
                 vocabulary: Vocabulary,
                 data_id: str,
                 layers: Optional[List[int]]=None,
                 activation: Callable[[tf.Tensor], tf.Tensor]=tf.tanh,
                 dropout_keep_prob: float=0.5,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        self.encoders = encoders
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.layers = layers
        self.activation = activation
        self.dropout_keep_prob = dropout_keep_prob
        self.max_output_len = 1

        with tf.variable_scope(name):
            self.learning_step = tf.get_variable(
                "learning_step", [], trainable=False,
                initializer=tf.constant_initializer(0))

            self.dropout_placeholder = \
                tf.placeholder(tf.float32, name="dropout_plc")
            self.gt_inputs = [tf.placeholder(
                tf.int32, shape=[None], name="targets")]
            mlp_input = tf.concat(1, [enc.encoded for enc in encoders])
            mlp = MultilayerPerceptron(
                mlp_input, layers, self.dropout_placeholder, len(vocabulary))

            self.loss_with_gt_ins = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    mlp.logits, self.gt_inputs[0]))
            self.loss_with_decoded_ins = self.loss_with_gt_ins
            self.cost = self.loss_with_gt_ins

            self.decoded_seq = [mlp.classification]
            self.decoded_logits = [mlp.logits]

            tf.scalar_summary(
                'val_optimization_cost', self.cost,
                collections=["summary_val"])
            tf.scalar_summary(
                'train_optimization_cost',
                self.cost, collections=["summary_train"])
    # pylint: enable=too-many-arguments

    @property
    def train_loss(self):
        return self.loss_with_gt_ins

    @property
    def runtime_loss(self):
        return self.loss_with_decoded_ins

    @property
    def decoded(self):
        return self.decoded_seq

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        sentences = cast(Iterable[List[str]],
                         dataset.get_series(self.data_id, allow_none=True))

        sentences_list = list(sentences) if sentences is not None else None

        fd = {}  # type: FeedDict

        label_tensors, _ = self.vocabulary.sentences_to_tensor(
            sentences_list, self.max_output_len)

        fd[self.gt_inputs[0]] = label_tensors[0]

        if train:
            fd[self.dropout_placeholder] = self.dropout_keep_prob
        else:
            fd[self.dropout_placeholder] = 1.0

        return fd
