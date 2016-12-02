"""This module contains helper functions that are suppoosed to be called from
the configuration file because calling the functions or the class constructors
directly would be inconvinent or impossible.
"""
#tests: lint
import tensorflow as tf

import neuralmonkey.vocabulary as vocabulary
import neuralmonkey.dataset as dataset

#pylint: disable=invalid-name
# for backwards compatibility
dataset_from_files = dataset.load_dataset_from_files
vocabulary_from_file = vocabulary.from_file
vocabulary_from_bpe = vocabulary.from_bpe
vocabulary_from_dataset = vocabulary.from_dataset
initialize_vocabulary = vocabulary.initialize_vocabulary

def adam_optimizer(learning_rate=1e-4):
    return tf.train.AdamOptimizer(learning_rate)

def adadelta_optimizer(**kwargs):
    return tf.train.AdadeltaOptimizer(**kwargs)

def configurable_type(base_type, **extra_kwargs):
    """Create a new "type" from a given type and additional arguments."""
    def new_type(*args, **kwargs):
        kwargs.update(extra_kwargs)
        return base_type(*args, **kwargs)

    return new_type
