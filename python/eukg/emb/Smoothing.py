import tensorflow as tf

# from .BaseModel import BaseModel


def add_semantic_network_loss(model):
  """
  Adds the semantic network loss to the graph
  :param model:
  :type model: BaseModel.BaseModel
  :return: the semantic network loss - 1D real valued tensor
  """
  print('Adding semantic network to graph')
  # dataset3: sr, ss, so, sns, sno, t, conc, counts
  model.smoothing_placeholders['sn_relations'] = rel = tf.placeholder(dtype=tf.int32, shape=[None], name='sn_relations')
  model.smoothing_placeholders['sn_pos_subj'] = psubj = tf.placeholder(dtype=tf.int32, shape=[None], name='sn_pos_subj')
  model.smoothing_placeholders['sn_pos_obj'] = pobj = tf.placeholder(dtype=tf.int32, shape=[None], name='sn_pos_obj')
  model.smoothing_placeholders['sn_neg_subj'] = nsubj = tf.placeholder(dtype=tf.int32, shape=[None], name='sn_neg_subj')
  model.smoothing_placeholders['sn_neg_obj'] = nobj = tf.placeholder(dtype=tf.int32, shape=[None], name='sn_neg_obj')
  model.smoothing_placeholders['sn_types'] = types = tf.placeholder(dtype=tf.int32, shape=[None], name='sn_types')
  model.smoothing_placeholders['sn_concepts'] = concepts = tf.placeholder(dtype=tf.int32,
                                                                          shape=[None, model.max_concepts_per_type],
                                                                          name='sn_concepts')
  model.smoothing_placeholders['sn_conc_counts'] = counts = tf.placeholder(dtype=tf.int32,
                                                                           shape=[None],
                                                                           name='sn_conc_counts')

  with tf.variable_scope("energy"):
    model.sn_pos_energy = pos_energy = model.embedding_model.energy(psubj, rel, pobj, model.energy_norm)
  with tf.variable_scope("energy", reuse=True):
    model.sn_neg_energy = neg_energy = model.embedding_model.energy(nsubj, rel, nobj, model.energy_norm)
  energy_loss = tf.reduce_mean(tf.nn.relu(model.gamma - neg_energy + pos_energy))
  model.sn_reward = -tf.reduce_mean(neg_energy, name='sn_reward')

  # [batch_size, embedding_size]
  type_embeddings = model.embedding_model.embedding_lookup(types)
  # [batch_size, max_concepts_per_type, embedding_size]
  concepts_embeddings = model.embedding_model.embedding_lookup(concepts)

  def calc_alignment_loss(_type_embeddings, _concept_embeddings):
    mask = tf.expand_dims(tf.sequence_mask(counts, maxlen=model.max_concepts_per_type, dtype=tf.float32), axis=-1)
    sum_ = tf.reduce_sum(mask * _concept_embeddings, axis=1, keepdims=False)
    float_counts = tf.to_float(tf.expand_dims(counts, axis=-1))
    avg_conc_embeddings = sum_ / tf.maximum(float_counts, tf.ones_like(float_counts))
    return tf.reduce_mean(tf.abs(_type_embeddings - avg_conc_embeddings))

  if isinstance(type_embeddings, tuple):
    alignment_loss = 0
    for type_embeddings_i, concepts_embeddings_i in zip(type_embeddings, concepts_embeddings):
      alignment_loss += calc_alignment_loss(type_embeddings_i, concepts_embeddings_i)
  else:
    alignment_loss = calc_alignment_loss(type_embeddings, concepts_embeddings)

  # summary
  tf.summary.scalar('sn_energy_loss', energy_loss)
  tf.summary.scalar('sn_alignment_loss', alignment_loss)
  # tf.summary.scalar('sn_accuracy', accuracy)
  avg_pos_energy = tf.reduce_mean(pos_energy)
  tf.summary.scalar('sn_pos_energy', avg_pos_energy)
  avg_neg_energy = tf.reduce_mean(neg_energy)
  tf.summary.scalar('sn_neg_energy', avg_neg_energy)

  # is this loss?
  loss = model.semnet_energy_param * energy_loss + model.semnet_alignment_param * alignment_loss
  return loss


def add_gen_semantic_network(model):
  """
  Adds the semantic network loss to the graph
  :param model:
  :type model: ..gan.Generator.Generator
  :return: the semantic network loss - 1D real valued tensor
  """
  print('Adding semantic network to graph')
  # dataset4: sr, ss, so, sns, sno, t, conc, counts
  model.smoothing_placeholders['sn_relations'] = rel = tf.placeholder(dtype=tf.int32, shape=[None], name='sn_relations')
  model.smoothing_placeholders['sn_pos_subj'] = psubj = tf.placeholder(dtype=tf.int32, shape=[None], name='sn_pos_subj')
  model.smoothing_placeholders['sn_pos_obj'] = pobj = tf.placeholder(dtype=tf.int32, shape=[None], name='sn_pos_obj')
  model.smoothing_placeholders['sn_neg_subj'] = nsubj = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                                                       name='sn_neg_subj')
  model.smoothing_placeholders['sn_neg_obj'] = nobj = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                                                     name='sn_neg_obj')
  model.smoothing_placeholders['sn_concepts'] = concepts = tf.placeholder(dtype=tf.int32,
                                                                          shape=[None, model.max_concepts_per_type],
                                                                          name='sn_concepts')
  model.smoothing_placeholders['sn_conc_counts'] = counts = tf.placeholder(dtype=tf.int32,
                                                                           shape=[None],
                                                                           name='sn_conc_counts')
  model.smoothing_placeholders['sn_types'] = types = tf.placeholder(dtype=tf.int32, shape=[None], name='sn_types')

  true_energy = model.embedding_model.energy(psubj, rel, pobj)
  sampl_energy = model.embedding_model.energy(nsubj, tf.expand_dims(rel, axis=-1), nobj, model.energy_norm)
  if model.gan_mode:
    model.sampl_distributions = tf.nn.softmax(-sampl_energy, axis=-1)
    model.type_probabilities = tf.gather_nd(model.sampl_distributions, model.gan_loss_sample, name='sampl_probs')
  else:
    sm_numerator = tf.exp(-true_energy)
    exp_sampl_energies = tf.exp(-sampl_energy)
    sm_denominator = tf.reduce_sum(exp_sampl_energies, axis=-1) + sm_numerator
    model.type_probabilities = sm_numerator / sm_denominator
  energy_loss = -tf.reduce_mean(tf.log(model.type_probabilities))

  # [batch_size, embedding_size]
  type_embeddings = model.embedding_model.embedding_lookup(types)
  # [batch_size, max_concepts_per_type, embedding_size]
  concepts_embeddings = model.embedding_model.embedding_lookup(concepts)

  def calc_alignment_loss(_type_embeddings, _concept_embeddings):
    mask = tf.expand_dims(tf.sequence_mask(counts, maxlen=model.max_concepts_per_type, dtype=tf.float32), axis=-1)
    sum_ = tf.reduce_sum(mask * _concept_embeddings, axis=1, keepdims=False)
    float_counts = tf.to_float(tf.expand_dims(counts, axis=-1))
    avg_conc_embeddings = sum_ / tf.maximum(float_counts, tf.ones_like(float_counts))
    return tf.reduce_mean(tf.abs(_type_embeddings - avg_conc_embeddings))

  if isinstance(type_embeddings, tuple):
    alignment_loss = 0
    for type_embeddings_i, concepts_embeddings_i in zip(type_embeddings, concepts_embeddings):
      alignment_loss += calc_alignment_loss(type_embeddings_i, concepts_embeddings_i)
  else:
    alignment_loss = calc_alignment_loss(type_embeddings, concepts_embeddings)

  return energy_loss, alignment_loss
