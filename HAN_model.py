import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import data_util
from model_components import task_specific_attention, bidirectional_rnn


class HANClassifierModel():
    """ Implementation of document classification model described in
      `Hierarchical Attention Networks for Document Classification (Yang et al., 2016)`
      (https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)"""

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 classes,
                 word_cell,
                 sentence_cell,
                 max_grad_norm,
                 dropout_keep_proba,
                 is_training=None,
                 learning_rate=1e-4,
                 device='/cpu:0',
                 scope=None):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.classes = classes
        self.word_cell = word_cell
        self.sentence_cell = sentence_cell
        self.max_grad_norm = max_grad_norm
        self.dropout_keep_proba = dropout_keep_proba

        with tf.variable_scope(scope or 'tcm') as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            if is_training is not None:
                self.is_training = is_training
            else:
                self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

            self.sample_weights = tf.placeholder(shape=(None,), dtype=tf.float32, name='sample_weights')

            # [document x sentence x word]
            self.inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='inputs')

            # [document x sentence]
            self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')

            # [document]
            self.sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')

            # [document]
            self.labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='labels')

            (self.document_size,
             self.sentence_size,
             self.word_size) = tf.unstack(tf.shape(self.inputs))

            with tf.device('/cpu:0'):
                self._init_embedding(scope)     # init self.embedding_matrix from self.inputs

            # embeddings cannot be placed on GPU
            with tf.device(device):
                self._init_body(scope)

        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

            self.loss = tf.reduce_mean(tf.multiply(self.cross_entropy, self.sample_weights))
            tf.summary.scalar('loss', self.loss)

            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

            tvars = tf.trainable_variables()

            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            # apply learning rate dacay here  2018.3.31
            self.learning_rate = tf.train.exponential_decay(learning_rate,
                                                       global_step=self.global_step,
                                                       decay_steps=200, decay_rate=0.9)
            self.learning_rate = tf.clip_by_value(self.learning_rate, clip_value_min=0.0001, clip_value_max=0.1)
            opt = tf.train.AdamOptimizer(self.learning_rate)

            self.train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)

            self.summary_op = tf.summary.merge_all()

    def _init_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.embedding_matrix = tf.get_variable(
                    name="embedding_matrix",
                    shape=[self.vocab_size, self.embedding_size],
                    initializer=layers.xavier_initializer(),
                    dtype=tf.float32)
                self.inputs_embedded = tf.nn.embedding_lookup(
                    self.embedding_matrix, self.inputs)     # lookup inputs(ids) in embedding_matrix to convert word from id to embedding vector

    def _init_body(self, scope):
        with tf.variable_scope(scope):
            word_level_inputs = tf.reshape(self.inputs_embedded, [
                self.document_size * self.sentence_size,
                self.word_size,
                self.embedding_size
            ])
            word_level_lengths = tf.reshape(
                self.word_lengths, [self.document_size * self.sentence_size])   # 2D(self.word_lengths) to 1D(word_level_lengths)

            with tf.variable_scope('word') as scope:
                word_encoder_output, _ = bidirectional_rnn(
                    self.word_cell, self.word_cell,
                    word_level_inputs, word_level_lengths,
                    scope=scope)

                with tf.variable_scope('attention') as scope:
                    word_level_output, words_attentions = task_specific_attention(
                        word_encoder_output,
                        scope=scope)
                    self.words_attentions = tf.reshape(words_attentions, shape=[self.document_size, self.sentence_size, self.word_size])

                with tf.variable_scope('dropout'):
                    word_level_output = layers.dropout(
                        word_level_output, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training,
                    )

            # sentence_level

            sentence_inputs = tf.reshape(
                word_level_output, [self.document_size, self.sentence_size, self.word_cell.output_size * 2])

            with tf.variable_scope('sentence') as scope:
                sentence_encoder_output, _ = bidirectional_rnn(
                    self.sentence_cell, self.sentence_cell, sentence_inputs, self.sentence_lengths, scope=scope)

                with tf.variable_scope('attention') as scope:
                    sentence_level_output, sentences_attentions = task_specific_attention(
                        sentence_encoder_output, scope=scope)
                    self.sentences_attentions = tf.reshape(sentences_attentions, shape=[self.document_size, self.sentence_size])

                with tf.variable_scope('dropout'):
                    sentence_level_output = layers.dropout(
                        sentence_level_output, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training,
                    )

            with tf.variable_scope('classifier'):
                self.logits = layers.fully_connected(
                    sentence_level_output, self.classes, activation_fn=None)

                self.prediction = tf.argmax(self.logits, axis=-1)

    def get_feed_data(self, x, y=None, class_weights=None, is_training=True):
        x_m, doc_sizes, sent_sizes = data_util.batch(x)
        fd = {
            self.inputs: x_m,
            self.sentence_lengths: doc_sizes,
            self.word_lengths: sent_sizes,
        }
        if y is not None:
            fd[self.labels] = y
            if class_weights is not None:
                fd[self.sample_weights] = [class_weights[yy] for yy in y]
            else:
                fd[self.sample_weights] = np.ones(shape=[len(x_m)], dtype=np.float32)
        fd[self.is_training] = is_training
        return fd
