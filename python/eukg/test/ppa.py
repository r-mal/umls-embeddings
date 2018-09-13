import tensorflow as tf
import numpy as np
import random
import os
import json
from tqdm import tqdm

from ..data import data_util, DataGenerator
from .. import Config, train

config = Config.flags


def evaluate():
  random.seed(1337)
  config.no_semantic_network = True

  cui2id, train_data, _, _ = data_util.load_metathesaurus_data(config.data_dir, config.val_proportion)
  id2cui = {v: k for k, v in cui2id.iteritems()}
  test_data = data_util.load_metathesaurus_test_data(config.data_dir)
  print('Loaded %d test triples from %s' % (len(test_data['rel']), config.data_dir))
  concept_ids = np.unique(np.concatenate([train_data['subj'], train_data['obj'], test_data['subj'], test_data['obj']]))
  print('%d total unique concepts' % len(concept_ids))
  data_generator = DataGenerator.DataGenerator(test_data,
                                               train_idx=np.arange(len(test_data['rel'])),
                                               val_idx=[],
                                               config=config,
                                               test_mode=True)

  valid_triples = set()
  for s, r, o in zip(train_data['subj'], train_data['rel'], train_data['obj']):
    valid_triples.add((s, r, o))
  for s, r, o in zip(test_data['subj'], test_data['rel'], test_data['obj']):
    valid_triples.add((s, r, o))
  print('%d valid triples' % len(valid_triples))

  model_name = config.run_name
  if config.mode == 'gan':
    scope = config.dis_run_name
    model_name += '/discriminator'
    config.mode = 'disc'
  else:
    scope = config.run_name

  with tf.Graph().as_default(), tf.Session() as session:
    # init model
    with tf.variable_scope(scope):
      model = train.init_model(config, data_generator)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # init saver
    tf_saver = tf.train.Saver(max_to_keep=10)

    # load model
    ckpt = tf.train.latest_checkpoint(os.path.join(config.model_dir, config.model, model_name))
    print('Loading checkpoint: %s' % ckpt)
    tf_saver.restore(session, ckpt)
    tf.get_default_graph().finalize()

    def decode_triple(s_, r_, o_, score):
      return id2cui[s_], id2cui[r_], id2cui[o_], str(score)

    incorrect = []  # list of tuples of (triple, corrupted_triple)
    num_correct = 0
    total = 0
    for r, s, o, ns, no in tqdm(data_generator.generate_mt(True), total=data_generator.num_train_batches()):
      pscores, nscores = session.run([model.pos_energy, model.neg_energy], {model.relations: r,
                                                                            model.pos_subj: s,
                                                                            model.pos_obj: o,
                                                                            model.neg_subj: ns,
                                                                            model.neg_obj: no})
      total += len(r)
      for pscore, nscore, rel, subj, obj, nsubj, nobj in zip(pscores, nscores, r, s, o, ns, no):
        if pscore < nscore:
          num_correct += 1
        else:
          incorrect.append((decode_triple(subj, rel, obj, pscore), decode_triple(nsubj, rel, nobj, nscore)))
    ppa = float(num_correct)/total
    print('PPA: %.4f' % ppa)
    outdir = os.path.join(config.eval_dir, config.run_name)
    if not os.path.exists(outdir):
      os.makedirs(outdir)
    json.dump(incorrect, open(os.path.join(outdir, 'ppa_incorrect.json'), 'w+'))
    with open(os.path.join(outdir, 'ppa.txt'), 'w+') as f:
      f.write(str(ppa))


def evaluate_sn():
  random.seed(1337)
  config.no_semantic_network = False

  data = {}
  cui2id, _, _, _ = data_util.load_metathesaurus_data(config.data_dir, 0.)
  id2cui = {v: k for k, v in cui2id.iteritems()}
  _ = data_util.load_semantic_network_data(config.data_dir, data)
  subj, rel, obj = data['sn_subj'], data['sn_rel'], data['sn_obj']
  print('Loaded %d sn triples from %s' % (len(rel), config.data_dir))

  valid_triples = set()
  for trip in zip(subj, rel, obj):
    valid_triples.add(trip)
  print('%d valid triples' % len(valid_triples))
  idxs = np.arange(len(rel))
  np.random.shuffle(idxs)
  idxs = idxs[:600]
  subj, rel, obj = subj[idxs], rel[idxs], obj[idxs]
  sampler = DataGenerator.NegativeSampler(valid_triples=valid_triples, name='???')
  nsubj, nobj = sampler.sample(subj, rel, obj)

  model_name = config.run_name
  if config.mode == 'gan':
    scope = config.dis_run_name
    model_name += '/discriminator'
    config.mode = 'disc'
  else:
    scope = config.run_name

  with tf.Graph().as_default(), tf.Session() as session:
    # init model
    with tf.variable_scope(scope):
      model = train.init_model(config, None)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # init saver
    tf_saver = tf.train.Saver(max_to_keep=10)

    # load model
    ckpt = tf.train.latest_checkpoint(os.path.join(config.model_dir, config.model, model_name))
    print('Loading checkpoint: %s' % ckpt)
    tf_saver.restore(session, ckpt)
    tf.get_default_graph().finalize()

    def decode_triple(s_, r_, o_, score):
      return id2cui[s_], id2cui[r_], id2cui[o_], str(score)

    incorrect = []  # list of tuples of (triple, corrupted_triple)
    num_correct = 0
    total = len(rel)
    feed_dict = {model.smoothing_placeholders['sn_relations']: rel,
                 model.smoothing_placeholders['sn_pos_subj']: subj,
                 model.smoothing_placeholders['sn_pos_obj']: obj,
                 model.smoothing_placeholders['sn_neg_subj']: nsubj,
                 model.smoothing_placeholders['sn_neg_obj']: nobj}
    pscores, nscores = session.run([model.sn_pos_energy, model.sn_neg_energy], feed_dict)
    for pscore, nscore, r, s, o, ns, no in zip(pscores, nscores, rel, subj, obj, nsubj, nobj):
      if pscore < nscore:
        num_correct += 1
      else:
        incorrect.append((decode_triple(s, r, o, pscore), decode_triple(ns, r, no, nscore)))
    ppa = float(num_correct)/total
    print('PPA: %.4f' % ppa)
    outdir = os.path.join(config.eval_dir, config.run_name)
    if not os.path.exists(outdir):
      os.makedirs(outdir)
    json.dump(incorrect, open(os.path.join(outdir, 'sn_ppa_incorrect.json'), 'w+'))
    with open(os.path.join(outdir, 'sn_ppa.txt'), 'w+') as f:
      f.write(str(ppa))


if __name__ == "__main__":
  if config.eval_mode == 'sn':
    evaluate_sn()
  else:
    evaluate()
