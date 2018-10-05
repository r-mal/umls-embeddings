import tensorflow as tf
import os
import math
import random
from tqdm import tqdm
import numpy as np
from itertools import izip

from .. import Config
from ..data import data_util, DataGenerator
from ..emb import EmbeddingModel
from Generator import GanGenerator
import Discriminator


# noinspection PyUnboundLocalVariable
def train():
  config = Config.flags

  use_semnet = not config.no_semantic_network

  # init model dir
  gan_model_dir = os.path.join(config.model_dir, config.model, config.run_name)
  if not os.path.exists(gan_model_dir):
    os.makedirs(gan_model_dir)

  # init summaries dir
  config.summaries_dir = os.path.join(config.summaries_dir, config.run_name)
  if not os.path.exists(config.summaries_dir):
    os.makedirs(config.summaries_dir)

  # save the config
  data_util.save_config(gan_model_dir, config)

  # load data
  cui2id, data, train_idx, val_idx = data_util.load_metathesaurus_data(config.data_dir, config.val_proportion)
  config.val_progress_update_interval = int(math.floor(float(len(val_idx)) / config.batch_size))
  config.batches_per_epoch = int(math.floor(float(len(train_idx)) / config.batch_size))
  if not config.no_semantic_network:
    type2cuis = data_util.load_semantic_network_data(config.data_dir, data)
  else:
    type2cuis = None
  data_generator = DataGenerator.DataGenerator(data, train_idx, val_idx, config, type2cuis)

  with tf.Graph().as_default(), tf.Session() as session:
    # init models
    with tf.variable_scope(config.dis_run_name):
      discriminator = init_model(config, 'disc')
    with tf.variable_scope(config.gen_run_name):
      config.no_semantic_network = True
      config.learning_rate = 1e-1
      generator = init_model(config, 'gen')
    if use_semnet:
      with tf.variable_scope(config.sn_gen_run_name):
        config.no_semantic_network = False
        sn_generator = init_model(config, 'sn_gen')

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # init saver
    dis_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=config.dis_run_name),
                               max_to_keep=10)
    gen_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=config.gen_run_name),
                               max_to_keep=10)

    # load models
    dis_ckpt = tf.train.latest_checkpoint(os.path.join(config.model_dir, config.model, config.dis_run_name))
    dis_saver.restore(session, dis_ckpt)
    gen_ckpt = tf.train.latest_checkpoint(os.path.join(config.model_dir, "distmult", config.gen_run_name))
    gen_saver.restore(session, gen_ckpt)
    if use_semnet:
      sn_gen_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                               scope=config.sn_gen_run_name),
                                    max_to_keep=10)
      sn_gen_ckpt = tf.train.latest_checkpoint(os.path.join(config.model_dir, "distmult", config.sn_gen_run_name))
      sn_gen_saver.restore(session, sn_gen_ckpt)

    # finalize graph
    tf.get_default_graph().finalize()

    # define streaming_accuracy reset per epoch
    print('local variables that will be reinitialized every epoch: %s' % tf.local_variables())
    reset_local_vars = lambda: session.run(discriminator.reset_streaming_metrics_op)

    # init summary directories and summary writers
    if not os.path.exists(os.path.join(config.summaries_dir, 'train')):
      os.makedirs(os.path.join(config.summaries_dir, 'train'))
    train_summary_writer = tf.summary.FileWriter(os.path.join(config.summaries_dir, 'train'))
    if not os.path.exists(os.path.join(config.summaries_dir, 'val')):
      os.makedirs(os.path.join(config.summaries_dir, 'val'))
    val_summary_writer = tf.summary.FileWriter(os.path.join(config.summaries_dir, 'val'))

    # config_map = config.flag_values_dict()
    # config_map['data'] = data
    # config_map['train_idx'] = train_idx
    # config_map['val_idx'] = val_idx

    global_step = 0
    for ep in xrange(config.num_epochs):
      print('----------------------------')
      print('Begin Train Epoch %d' % ep)
      if use_semnet:
        global_step = train_epoch_sn(session, discriminator, generator, sn_generator, config, data_generator,
                                     train_summary_writer, global_step)
        sn_gen_saver.save(session, os.path.join(gan_model_dir, 'sn_generator', config.model), global_step=global_step)
      else:
        global_step = train_epoch(session, discriminator, generator, config, data_generator, train_summary_writer,
                                  global_step)
      print("Saving models to %s at step %d" % (gan_model_dir, global_step))
      dis_saver.save(session, os.path.join(gan_model_dir, 'discriminator', config.model), global_step=global_step)
      gen_saver.save(session, os.path.join(gan_model_dir, 'generator', config.model), global_step=global_step)
      reset_local_vars()
      print('----------------------------')
      print('Begin Validation Epoch %d' % ep)
      validation_epoch(session, discriminator, config, data_generator, val_summary_writer, global_step)


def init_model(config, mode):
  print('Initializing %s model...' % mode)

  if mode == 'disc':
    if config.model == 'transe':
      em = EmbeddingModel.TransE(config)
    elif config.model == 'transd':
      # config.embedding_size = config.embedding_size / 2
      em = EmbeddingModel.TransD(config)
      # config.embedding_size = config.embedding_size * 2
    else:
      raise ValueError('Unrecognized model type: %s' % config.model)
    model = Discriminator.BaseModel(config, em)
  elif mode == 'gen':
    em = EmbeddingModel.DistMult(config)
    model = GanGenerator(config, em)
  elif mode == 'sn_gen':
    em = EmbeddingModel.DistMult(config)
    model = GanGenerator(config, em)
  else:
    raise ValueError('Unrecognized mode: %s' % config.mode)

  model.build()
  return model


def find_unique(tensor_list):
  if max([len(t.shape) for t in tensor_list[:10]]) == 1:
    return np.unique(np.concatenate(tensor_list[:10]))
  else:
    return np.unique(np.concatenate([t.flatten() for t in tensor_list[:10]]))


def sample_corrupted_triples(sampl_sub, sampl_obj, probability_distributions, idx_np):
  nsub = []
  nobj = []
  sampl_idx = []
  for i, dist in enumerate(probability_distributions):
    [j] = np.random.choice(idx_np, [1], p=dist)
    nsub.append(sampl_sub[i, j])
    nobj.append(sampl_obj[i, j])
    sampl_idx.append([i, j])
  nsub = np.asarray(nsub)
  nobj = np.asarray(nobj)
  return nsub, nobj, sampl_idx


def train_epoch(session, discriminator, generator, config, data_generator, summary_writer, global_step):
  baseline = 0.
  console_update_interval = config.progress_update_interval
  pbar = tqdm(total=console_update_interval)
  idx_np = np.arange(config.num_generator_samples)
  for b, batch in enumerate(data_generator.generate_mt_gen_mode(True)):
    verbose_batch = b > 0 and b % console_update_interval == 0

    # generation
    gen_feed_dict = generator.prepare_feed_dict(batch, True)
    probability_distributions = session.run(generator.probability_distributions, gen_feed_dict)
    rel, psub, pobj, sampl_sub, sampl_obj = batch
    nsub = []
    nobj = []
    sampl_idx = []
    for i, dist in enumerate(probability_distributions):
      [j] = np.random.choice(idx_np, [1], p=dist)
      nsub.append(sampl_sub[i, j])
      nobj.append(sampl_obj[i, j])
      sampl_idx.append([i, j])
    nsub = np.asarray(nsub)
    nobj = np.asarray(nobj)

    # discrimination
    dis_fetched = session.run(discriminator.fetches(True, verbose_batch) + [discriminator.reward],
                              {discriminator.relations: rel,
                               discriminator.pos_subj: psub,
                               discriminator.pos_obj: pobj,
                               discriminator.neg_subj: nsub,
                               discriminator.neg_obj: nobj})

    # generation reward
    discounted_reward = dis_fetched[-1] - baseline
    baseline = dis_fetched[-1]
    gen_feed_dict[generator.discounted_reward] = discounted_reward
    gen_feed_dict[generator.gan_loss_sample] = np.asarray(sampl_idx)
    gen_fetched = session.run([generator.summary, generator.loss, generator.probabilities, generator.train_op],
                              gen_feed_dict)
    # assert gloss == gen_fetched[1], \
    #   "Forward pass for generation step does not match forward pass for generator learning! %f != %f" \
    #     % (gloss, gen_fetched[1])

    # update tensorboard summary
    summary_writer.add_summary(dis_fetched[0], global_step)
    summary_writer.add_summary(gen_fetched[0], global_step)
    global_step += 1

    # perform normalization
    session.run([generator.norm_op, discriminator.norm_op],
                {generator.ids_to_update: find_unique(batch),
                 discriminator.ids_to_update: find_unique([rel, psub, pobj, nsub, nobj])})

    # udpate progress bar
    pbar.set_description("Training Batch: %d. GLoss: %.4f. DLoss: %.4f. Reward: %.4f" %
                         (b, gen_fetched[1], dis_fetched[1], discounted_reward))
    pbar.update()

    if verbose_batch:
      print('Discriminator:')
      discriminator.progress_update(batch, dis_fetched)
      print('Generator:')
      print('Avg probability of sampled negative examples from last batch: %.4f' % np.average(gen_fetched[2]))
      pbar.close()
      pbar = tqdm(total=console_update_interval)
  pbar.close()

  return global_step


def train_epoch_sn(sess, discriminator, generator, sn_generator, config, data_generator, summary_writer, global_step):
  baseline = 0.
  sn_baseline = 0.
  pbar = None
  console_update_interval = config.progress_update_interval
  idx_np = np.arange(config.num_generator_samples)
  sn_idx_np = np.arange(10)
  for b, (mt_batch, sn_batch) in enumerate(izip(data_generator.generate_mt_gen_mode(True),
                                                data_generator.generate_sn_gen_mode(True))):
    verbose_batch = b > 0 and b % console_update_interval == 0

    # mt generation
    gen_feed_dict = generator.prepare_feed_dict(mt_batch, True)
    probability_distributions = sess.run(generator.probability_distributions, gen_feed_dict)
    rel, psub, pobj, sampl_sub, sampl_obj = mt_batch
    nsub, nobj, sampl_idx = sample_corrupted_triples(sampl_sub, sampl_obj, probability_distributions, idx_np)

    # sn generation
    sn_gen_feed_dict = {sn_generator.smoothing_placeholders['sn_relations']: sn_batch[0],
                        sn_generator.smoothing_placeholders['sn_neg_subj']: sn_batch[3],
                        sn_generator.smoothing_placeholders['sn_neg_obj']: sn_batch[4]}
    type_distributions = sess.run(sn_generator.sampl_distributions, sn_gen_feed_dict)
    sn_nsub, sn_nobj, sn_sampl_idx = sample_corrupted_triples(sn_batch[3], sn_batch[4], type_distributions, sn_idx_np)
    types = np.unique(np.concatenate([sn_batch[1], sn_batch[2], sn_nsub, sn_nobj]))
    concepts = np.zeros([len(types), config.max_concepts_per_type], dtype=np.int32)
    concept_lens = np.zeros([len(types)], dtype=np.int32)
    for i, tid in enumerate(types):
      concepts_of_type_t = data_generator.type2cuis[tid] if tid in data_generator.type2cuis else []
      random.shuffle(concepts_of_type_t)
      concepts_of_type_t = concepts_of_type_t[:config.max_concepts_per_type]
      concept_lens[i] = len(concepts_of_type_t)
      concepts[i, :len(concepts_of_type_t)] = concepts_of_type_t

    # discrimination
    dis_fetched = sess.run(discriminator.fetches(True, verbose_batch) + [discriminator.sn_reward, discriminator.reward],
                           {discriminator.relations: rel,
                               discriminator.pos_subj: psub,
                               discriminator.pos_obj: pobj,
                               discriminator.neg_subj: nsub,
                               discriminator.neg_obj: nobj,
                               discriminator.smoothing_placeholders['sn_relations']: sn_batch[0],
                               discriminator.smoothing_placeholders['sn_pos_subj']: sn_batch[1],
                               discriminator.smoothing_placeholders['sn_pos_obj']: sn_batch[2],
                               discriminator.smoothing_placeholders['sn_neg_subj']: sn_nsub,
                               discriminator.smoothing_placeholders['sn_neg_obj']: sn_nobj,
                               discriminator.smoothing_placeholders['sn_types']: types,
                               discriminator.smoothing_placeholders['sn_concepts']: concepts,
                               discriminator.smoothing_placeholders['sn_conc_counts']: concept_lens})

    # generation reward
    discounted_reward = dis_fetched[-1] - baseline
    baseline = dis_fetched[-1]
    gen_feed_dict[generator.discounted_reward] = discounted_reward
    gen_feed_dict[generator.gan_loss_sample] = np.asarray(sampl_idx)
    gen_fetched = sess.run([generator.summary, generator.loss, generator.probabilities, generator.train_op],
                           gen_feed_dict)

    # sn generation reward
    sn_discounted_reward = dis_fetched[-2] - sn_baseline
    sn_baseline = dis_fetched[-2]
    sn_gen_feed_dict[sn_generator.discounted_reward] = sn_discounted_reward
    sn_gen_feed_dict[sn_generator.gan_loss_sample] = np.asarray(sn_sampl_idx)
    sn_gen_fetched = sess.run([sn_generator.summary, sn_generator.loss,
                               sn_generator.type_probabilities, sn_generator.train_op],
                              sn_gen_feed_dict)

    # update tensorboard summary
    summary_writer.add_summary(dis_fetched[0], global_step)
    summary_writer.add_summary(gen_fetched[0], global_step)
    summary_writer.add_summary(sn_gen_fetched[0], global_step)
    global_step += 1

    # perform normalization
    sess.run([generator.norm_op, discriminator.norm_op, sn_generator.norm_op],
             {generator.ids_to_update: find_unique(mt_batch + sn_batch),
                 discriminator.ids_to_update: find_unique([rel, psub, pobj, nsub, nobj]),
                 sn_generator.ids_to_update: sn_batch[7]})

    # udpate progress bar
    pbar = tqdm(total=console_update_interval) if pbar is None else pbar
    pbar.set_description("Training Batch: %d. GLoss: %.4f. SN_GLoss: %.4f. DLoss: %.4f." %
                         (b, gen_fetched[1], sn_gen_fetched[1], dis_fetched[1]))
    pbar.update()

    if verbose_batch:
      print('Discriminator:')
      discriminator.progress_update(mt_batch, dis_fetched)
      print('Generator:')
      print('Avg probability of sampled negative examples from last batch: %.4f' % np.average(gen_fetched[2]))
      print('SN Generator:')
      print('Avg probability of sampled negative examples from last batch: %.4f' % np.average(sn_gen_fetched[2]))
      pbar.close()
      pbar = tqdm(total=console_update_interval)
  if pbar:
    pbar.close()

  return global_step


def validation_epoch(session, model, config, data_generator, summary_writer, global_step):
  console_update_interval = config.val_progress_update_interval
  pbar = tqdm(total=console_update_interval)
  # validation epoch
  for b, batch in enumerate(data_generator.generate_mt(False)):
    verbose_batch = b > 0 and b % console_update_interval == 0

    fetched = session.run(model.fetches(False, verbose=verbose_batch), model.prepare_feed_dict(batch, False))

    # update tensorboard summary
    summary_writer.add_summary(fetched[0], global_step)
    global_step += 1

    # udpate progress bar
    pbar.set_description("Validation Batch: %d. Loss: %.4f" % (b, fetched[1]))
    pbar.update()

    if verbose_batch:
      model.progress_update(batch, fetched)
      pbar.close()
      pbar = tqdm(total=console_update_interval)
  pbar.close()
