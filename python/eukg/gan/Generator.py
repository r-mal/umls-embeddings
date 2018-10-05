import tensorflow as tf
import numpy as np

from Discriminator import BaseModel
from ..emb import Smoothing
from ..data import DataGenerator


class Generator(BaseModel):
  def __init__(self, config, embedding_model, data_generator=None):
    BaseModel.__init__(self, config, embedding_model, data_generator)
    self.num_samples = config.num_generator_samples
    self.gan_mode = False

    # dataset2: s, r, o, ns, no
    self.neg_subj = tf.placeholder(dtype=tf.int32, shape=[None, self.num_samples], name="neg_subj")
    self.neg_obj = tf.placeholder(dtype=tf.int32, shape=[None, self.num_samples], name="neg_obj")
    self.discounted_reward = tf.placeholder(dtype=tf.float32, shape=[], name="discounted_reward")
    self.gan_loss_sample = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="gan_loss_sample")

    self.probabilities = None
    self.sampl_energies = None
    self.true_energies = None
    self.probability_distributions = None

    # semantic network vars
    self.type_probabilities = None

  def build(self):
    summary = []
    # [batch_size, num_samples]
    self.sampl_energies = self.embedding_model.energy(self.neg_subj,
                                                      tf.expand_dims(self.relations, axis=1),
                                                      self.neg_obj)
    # [batch_size]
    self.true_energies = self.embedding_model.energy(self.pos_subj,
                                                     self.relations,
                                                     self.pos_obj)

    # backprop
    optimizer = self.optimizer()
    # [batch_size]
    sm_numerator = tf.exp(-self.true_energies)
    # [batch_size]
    exp_sampl_nergies = tf.exp(-self.sampl_energies)
    sm_denominator = tf.reduce_sum(exp_sampl_nergies, axis=-1) + sm_numerator
    # [batch_size]
    self.probabilities = sm_numerator / sm_denominator
    self.loss = -tf.reduce_mean(tf.log(self.probabilities))

    # regularization for distmult
    if self.model == "distmult":
      reg = self.regulatization_parameter * self.embedding_model.regularization([self.pos_subj, self.pos_obj,
                                                                                 self.neg_subj, self.neg_obj,
                                                                                 self.relations])
      summary += [tf.summary.scalar('reg', reg),
                  tf.summary.scalar('log_prob', self.loss)]
      self.loss += reg

    if self.use_semantic_network:
      sn_energy_loss, sn_alignment_loss = Smoothing.add_gen_semantic_network(self)
      self.loss += self.semnet_energy_param * sn_energy_loss + self.semnet_alignment_param * sn_alignment_loss
      summary += [tf.summary.scalar('sn_energy_loss', sn_energy_loss / self.batch_size),
                  tf.summary.scalar('sn_alignment_loss', sn_alignment_loss / self.batch_size)]

    self.avg_pos_energy = tf.reduce_mean(self.true_energies)
    self.avg_neg_energy = tf.reduce_mean(self.sampl_energies)
    summary += [tf.summary.scalar('loss', self.loss),
                tf.summary.scalar('avg_prob', tf.reduce_mean(self.probabilities)),
                tf.summary.scalar('min_prob', tf.reduce_min(self.probabilities)),
                tf.summary.scalar('max_prob', tf.reduce_max(self.probabilities)),
                tf.summary.scalar('pos_energy', self.avg_pos_energy),
                tf.summary.scalar('neg_energy', self.avg_neg_energy),
                tf.summary.scalar('margin', self.avg_pos_energy - self.avg_neg_energy)]
    self.train_op = optimizer.minimize(self.loss, tf.train.get_or_create_global_step())

    # summary
    self.summary = tf.summary.merge(summary)

  def fetches(self, is_training, verbose=False):
    fetches = [self.summary, self.loss]
    if verbose:
      fetches += [self.probabilities, self.avg_pos_energy, self.avg_neg_energy]
    if is_training:
      fetches += [self.train_op]
    return fetches

  def prepare_feed_dict(self, batch, is_training, **kwargs):
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
                self.smoothing_placeholders['sn_neg_subj']: [[0]],
                self.smoothing_placeholders['sn_neg_obj']: [[0]],
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
    print('Avg probability of last batch: %.4f' % np.average(fetched[2]))
    print('Avg pos energy of last batch: %.4f' % np.average(fetched[3]))
    print('Avg neg energy of last batch: %.4f' % np.average(fetched[4]))

  def data_provider(self, config, is_training, **kwargs):
    if self.use_semantic_network:
      return DataGenerator.wrap_generators(self.data_generator.generate_mt_gen_mode,
                                           self.data_generator.generate_sn_gen_mode, is_training)
    else:
      return self.data_generator.generate_mt_gen_mode(is_training)


class GanGenerator(Generator):
  def __init__(self, config, embedding_model, data_generator=None):
    Generator.__init__(self, config, embedding_model, data_generator)
    self.gan_mode = True
    self.sampl_distributions = None

  def build(self):
    # [batch_size, num_samples]
    self.sampl_energies = self.embedding_model.energy(self.neg_subj,
                                                      tf.expand_dims(self.relations, axis=1),
                                                      self.neg_obj)
    # [batch_size]
    self.true_energies = self.embedding_model.energy(self.pos_subj,
                                                     self.relations,
                                                     self.pos_obj)

    optimizer = self.optimizer()
    if self.use_semantic_network:
      # this method also adds values for self.sampl_distributions and self.type_probabilities
      loss, _ = Smoothing.add_gen_semantic_network(self)
      grads_and_vars = optimizer.compute_gradients(loss)
      vars_with_grad = [v for g, v in grads_and_vars if g is not None]
      if not vars_with_grad:
        raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))
      discounted_grads_and_vars = [(self.discounted_reward * g, v) for g, v in grads_and_vars if g is not None]
      self.train_op = optimizer.apply_gradients(discounted_grads_and_vars,
                                                global_step=tf.train.get_or_create_global_step())
      summary = [tf.summary.scalar('avg_st_prob', tf.reduce_mean(self.type_probabilities)),
                 tf.summary.scalar('sn_loss', loss / self.batch_size),
                 tf.summary.scalar('reward', self.discounted_reward)]
    else:
      # [batch_size, num_samples] - this is for sampling during GAN training
      self.probability_distributions = tf.nn.softmax(self.sampl_energies, axis=-1)
      self.probabilities = tf.gather_nd(self.probability_distributions, self.gan_loss_sample, name='sampl_probs')
      loss = -tf.reduce_sum(tf.log(self.probabilities))
      summary = [tf.summary.scalar('avg_sampled_prob', tf.reduce_mean(self.probabilities))]

      # if training as part of a GAN, gradients should be scaled by discounted_reward
      grads_and_vars = optimizer.compute_gradients(loss)
      vars_with_grad = [v for g, v in grads_and_vars if g is not None]
      if not vars_with_grad:
        raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))
      discounted_grads_and_vars = [(self.discounted_reward * g, v) for g, v in grads_and_vars if g is not None]
      self.train_op = optimizer.apply_gradients(discounted_grads_and_vars,
                                                global_step=tf.train.get_or_create_global_step())

    # reporting loss
    self.loss = loss / self.batch_size
    # summary
    self.summary = tf.summary.merge(summary)
