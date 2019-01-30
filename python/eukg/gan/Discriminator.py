import tensorflow as tf
import numpy as np

from ..data import DataGenerator
from ..emb import Smoothing

from tf_util import Trainable


class BaseModel(Trainable):
  def __init__(self, config, embedding_model, data_generator=None):
    """
    :param config: config map
    :param embedding_model: KG Embedding Model
    :type embedding_model: EmbeddingModel.BaseModel
    :param data_generator: data generator
    :type data_generator: DataGenerator.DataGenerator
    """
    Trainable.__init__(self)
    self.model = config.model
    self.embedding_model = embedding_model
    self.data_generator = data_generator

    # class variable declarations
    self.batch_size = config.batch_size
    # self.embedding_size = config.embedding_size
    self.vocab_size = config.vocab_size
    self.gamma = config.gamma
    self.embedding_device = config.embedding_device
    self.max_concepts_per_type = config.max_concepts_per_type
    # self.is_training = is_training
    self.energy_norm = config.energy_norm_ord
    self.use_semantic_network = not config.no_semantic_network
    self.semnet_alignment_param = config.semnet_alignment_param
    self.semnet_energy_param = config.semnet_energy_param
    self.regulatization_parameter = config.regularization_parameter

    # optimization
    self.learning_rate = tf.train.exponential_decay(config.learning_rate,
                                                    tf.train.get_or_create_global_step(),
                                                    config.batches_per_epoch,
                                                    config.decay_rate,
                                                    staircase=True)
    if config.optimizer == "adam":
      self.optimizer = lambda: tf.train.AdamOptimizer(self.learning_rate)
    else:
      self.optimizer = lambda: tf.train.MomentumOptimizer(self.learning_rate,
                                                          config.momentum,
                                                          use_nesterov=True)

    # input placeholders
    self.relations = tf.placeholder(dtype=tf.int32, shape=[None], name='relations')
    self.pos_subj = tf.placeholder(dtype=tf.int32, shape=[None], name='pos_subj')
    self.pos_obj = tf.placeholder(dtype=tf.int32, shape=[None], name='pos_obj')
    self.neg_subj = tf.placeholder(dtype=tf.int32, shape=[None], name='neg_subj')
    self.neg_obj = tf.placeholder(dtype=tf.int32, shape=[None], name='neg_obj')
    self.labels = tf.ones_like(self.relations, dtype=tf.int32)
    self.smoothing_placeholders = {}

    self.pos_energy = None
    self.neg_energy = None
    self.predictions = None
    self.reward = None
    self.sn_reward = None
    self.loss = None
    self.streaming_accuracy = None
    self.accuracy = None
    self.avg_pos_energy = None
    self.avg_neg_energy = None
    self.summary = None
    self.train_op = None

    # define reset op (for resetting counts for streaming metrics after each validation epoch)
    self.reset_streaming_metrics_op = tf.variables_initializer(tf.local_variables())

    # define norm op
    self.norm_op = self.embedding_model.normalize_parameters()
    self.ids_to_update = self.embedding_model.ids_to_update

  def build(self):
    # energies
    with tf.variable_scope("energy"):
      self.pos_energy = self.embedding_model.energy(self.pos_subj, self.relations, self.pos_obj, self.energy_norm)
    with tf.variable_scope("energy", reuse=True):
      self.neg_energy = self.embedding_model.energy(self.neg_subj, self.relations, self.neg_obj, self.energy_norm)
    self.predictions = tf.argmax(tf.stack([self.pos_energy, self.neg_energy], axis=1), axis=1, output_type=tf.int32)
    self.reward = tf.reduce_mean(self.neg_energy, name='reward')

    # loss
    self.loss = tf.reduce_mean(tf.nn.relu(self.gamma - self.neg_energy + self.pos_energy), name='loss')

    if self.model == 'distmult':
      reg = self.regulatization_parameter * self.embedding_model.regularization([self.pos_subj, self.pos_obj,
                                                                                 self.neg_subj, self.neg_obj,
                                                                                 self.relations])
      tf.summary.scalar('reg', reg)
      tf.summary.scalar('margin_loss', self.loss)
      self.loss += reg

    if self.use_semantic_network:
      semnet_loss = Smoothing.add_semantic_network_loss(self)
      self.loss += semnet_loss
      tf.summary.scalar('sn_loss', semnet_loss)
    # backprop
    self.train_op = self.optimizer().minimize(self.loss, tf.train.get_or_create_global_step())

    # summary
    tf.summary.scalar('loss', self.loss)
    _, self.streaming_accuracy = tf.metrics.accuracy(labels=self.labels, predictions=self.predictions)
    tf.summary.scalar('streaming_accuracy', self.streaming_accuracy)
    self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.predictions, self.labels)))
    tf.summary.scalar('accuracy', self.accuracy)
    self.avg_pos_energy = tf.reduce_mean(self.pos_energy)
    tf.summary.scalar('pos_energy', self.avg_pos_energy)
    self.avg_neg_energy = tf.reduce_mean(self.neg_energy)
    tf.summary.scalar('neg_energy', self.avg_neg_energy)
    tf.summary.scalar('margin', self.avg_neg_energy - self.avg_pos_energy)
    self.summary = tf.summary.merge_all()

  def fetches(self, is_training, verbose=False):
    fetches = [self.summary, self.loss]
    if verbose:
      if is_training:
        fetches += [self.accuracy]
      else:
        fetches += [self.streaming_accuracy]
      fetches += [self.avg_pos_energy, self.avg_neg_energy]
    if is_training:
      fetches += [self.train_op]
    return fetches

  def prepare_feed_dict(self, batch, is_training, **kwargs):
    # return {}
    if self.use_semantic_network:
      if is_training:
        rel, psub, pobj, nsub, nobj, sn_rel, sn_psub, sn_pobj, sn_nsub, sn_nobj, conc, c_lens, types = batch
        return {self.relations: rel,
                self.pos_subj: psub,
                self.pos_obj: pobj,
                self.neg_subj: nsub,
                self.neg_obj: nobj,
                self.smoothing_placeholders['sn_relations']: sn_rel,
                self.smoothing_placeholders['sn_pos_subj']: sn_psub,
                self.smoothing_placeholders['sn_pos_obj']: sn_pobj,
                self.smoothing_placeholders['sn_neg_subj']: sn_nsub,
                self.smoothing_placeholders['sn_neg_obj']: sn_nobj,
                self.smoothing_placeholders['sn_concepts']: conc,
                self.smoothing_placeholders['sn_conc_counts']: c_lens,
                self.smoothing_placeholders['sn_types']: types}
      else:
        rel, psub, pobj, nsub, nobj = batch
        return {self.relations: rel,
                self.pos_subj: psub,
                self.pos_obj: pobj,
                self.neg_subj: nsub,
                self.neg_obj: nobj,
                self.smoothing_placeholders['sn_relations']: [0],
                self.smoothing_placeholders['sn_pos_subj']: [0],
                self.smoothing_placeholders['sn_pos_obj']: [0],
                self.smoothing_placeholders['sn_neg_subj']: [0],
                self.smoothing_placeholders['sn_neg_obj']: [0],
                self.smoothing_placeholders['sn_concepts']: np.zeros([1, 1000], dtype=np.int32),
                self.smoothing_placeholders['sn_conc_counts']: [1],
                self.smoothing_placeholders['sn_types']: [0]}
    else:
      rel, psub, pobj, nsub, nobj = batch
      return {self.relations: rel,
              self.pos_subj: psub,
              self.pos_obj: pobj,
              self.neg_subj: nsub,
              self.neg_obj: nobj}

  def progress_update(self, batch, fetched, **kwargs):
    print('Avg loss of last batch: %.4f' % np.average(fetched[1]))
    print('Accuracy of last batch: %.4f' % np.average(fetched[2]))
    print('Avg pos energy of last batch: %.4f' % np.average(fetched[3]))
    print('Avg neg energy of last batch: %.4f' % np.average(fetched[4]))

  def data_provider(self, config, is_training, **kwargs):
    if self.use_semantic_network:
      return DataGenerator.wrap_generators(self.data_generator.generate_mt,
                                           self.data_generator.generate_sn, is_training)
    else:
      return self.data_generator.generate_mt(is_training)
