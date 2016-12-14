import numpy as np
import tensorflow as tf

# tests: mypy

class WordAlignmentDecoder(object):
    """
    A decoder that computes soft alignment from an attentive encoder. Loss is
    computed as cross-entropy against a reference alignment.
    """

    def __init__(self, encoder, decoder, data_id, name):
        self.encoder = encoder
        self.decoder = decoder
        self.data_id = data_id
        self.name = name

        self.ref_alignment = tf.placeholder(tf.float32, [None, None, None],
                                            name="ref_alignment")

        _, self.train_loss = self._make_decoder(runtime_mode=False)
        self.decoded, self.runtime_loss = self._make_decoder(runtime_mode=True)


    def _make_decoder(self, runtime_mode=False):
        attn_obj = self.decoder.get_attention_object(self.encoder,
                                                     runtime_mode,
                                                     create=False)

        alignment_logits = tf.transpose(tf.pack(attn_obj.logits_in_time),
                                        perm=[1, 0, 2])
        # alignment_logits = alignment_logits[:, :-1]  # last output is </s>


        if runtime_mode:
            alignment = tf.nn.softmax(alignment_logits)
            loss = 0
        else:
            alignment = None
            loss = tf.nn.softmax_cross_entropy_with_logits(alignment_logits,
                                                           self.ref_alignment)

        return alignment, loss

    @property
    def cost(self):
        return self.train_loss

    # pylint: disable=unused-argument
    def feed_dict(self, dataset, train=False):
        # pylint: disable=invalid-name
        # fd is the common name for feed dictionary
        fd = {}

        alignment = dataset.get_series(self.data_id, allow_none=True)
        if alignment is None:
            alignment = np.zeros((len(dataset),
                                  self.decoder.max_output_len,
                                  self.encoder.max_input_len), np.float32)

        fd[self.ref_alignment] = alignment

        return fd
