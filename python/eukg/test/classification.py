import tensorflow as tf
import numpy as np
import random
import os
from tqdm import tqdm

from ..data import data_util, DataGenerator
from .. import Config, train
from sklearn.svm import LinearSVC
from sklearn import metrics

config = Config.flags


def evaluate():
  random.seed(1337)
  np.random.seed(1337)
  config.no_semantic_network = True
  config.batch_size = 2000

  cui2id, train_data, _, _ = data_util.load_metathesaurus_data(config.data_dir, config.val_proportion)
  test_data = data_util.load_metathesaurus_test_data(config.data_dir)
  print('Loaded %d test triples from %s' % (len(test_data['rel']), config.data_dir))
  concept_ids = np.unique(np.concatenate([train_data['subj'], train_data['obj'], test_data['subj'], test_data['obj']]))
  print('%d total unique concepts' % len(concept_ids))
  val_idx = np.random.permutation(np.arange(len(train_data['rel'])))[:100000]
  val_data_generator = DataGenerator.DataGenerator(train_data,
                                                   train_idx=val_idx,
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

    scores = []
    labels = []
    for r, s, o, ns, no in tqdm(val_data_generator.generate_mt(True), total=val_data_generator.num_train_batches()):
      pscores, nscores = session.run([model.pos_energy, model.neg_energy], {model.relations: r,
                                                                            model.pos_subj: s,
                                                                            model.pos_obj: o,
                                                                            model.neg_subj: ns,
                                                                            model.neg_obj: no})
      scores += pscores.tolist()
      labels += np.ones_like(pscores, dtype=np.int).tolist()
      scores += nscores.tolist()
      labels += np.zeros_like(nscores, dtype=np.int).tolist()
    print('Calculated scores. Training SVM.')
    svm = LinearSVC(dual=False)
    svm.fit(np.asarray(scores).reshape(-1, 1), labels)
    print('Done.')

    data_generator = DataGenerator.DataGenerator(test_data,
                                                 train_idx=np.arange(len(test_data['rel'])),
                                                 val_idx=[],
                                                 config=config,
                                                 test_mode=True)
    data_generator._sampler = val_data_generator.sampler
    scores, labels = [], []
    for r, s, o, ns, no in tqdm(data_generator.generate_mt(True), desc='classifying',
                                total=data_generator.num_train_batches()):
      pscores, nscores = session.run([model.pos_energy, model.neg_energy], {model.relations: r,
                                                                            model.pos_subj: s,
                                                                            model.pos_obj: o,
                                                                            model.neg_subj: ns,
                                                                            model.neg_obj: no})
      scores += pscores.tolist()
      labels += np.ones_like(pscores, dtype=np.int).tolist()
      scores += nscores.tolist()
      labels += np.zeros_like(nscores, dtype=np.int).tolist()
    predictions = svm.predict(np.asarray(scores).reshape(-1, 1))
    print('pred: %s' % predictions.shape)
    print('lbl: %d' % len(labels))
    print('Relation Triple Classification Accuracy:  %.4f' % metrics.accuracy_score(labels, predictions))
    print('Relation Triple Classification Precision: %.4f' % metrics.precision_score(labels, predictions))
    print('Relation Triple Classification Recall:    %.4f' % metrics.recall_score(labels, predictions))
    print(metrics.classification_report(labels, predictions))


def evaluate_sn():
  random.seed(1337)
  config.no_semantic_network = False

  data = {}
  cui2id, _, _, _ = data_util.load_metathesaurus_data(config.data_dir, 0.)
  _ = data_util.load_semantic_network_data(config.data_dir, data)
  subj, rel, obj = data['sn_subj'], data['sn_rel'], data['sn_obj']
  print('Loaded %d sn triples from %s' % (len(rel), config.data_dir))

  valid_triples = set()
  for trip in zip(subj, rel, obj):
    valid_triples.add(trip)
  print('%d valid triples' % len(valid_triples))
  idxs = np.random.permutation(np.arange(len(rel)))
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

    feed_dict = {model.smoothing_placeholders['sn_relations']: rel,
                 model.smoothing_placeholders['sn_pos_subj']: subj,
                 model.smoothing_placeholders['sn_pos_obj']: obj,
                 model.smoothing_placeholders['sn_neg_subj']: nsubj,
                 model.smoothing_placeholders['sn_neg_obj']: nobj}
    pscores, nscores = session.run([model.sn_pos_energy, model.sn_neg_energy], feed_dict)
    scores = np.concatenate((pscores, nscores))
    labels = np.concatenate((np.ones_like(pscores, dtype=np.int), np.zeros_like(nscores, dtype=np.int)))
    print('Calculated scores. Training SVM.')
    svm = LinearSVC(dual=False)
    svm.fit(scores.reshape(-1, 1), labels)
    print('Done.')

    with np.load(os.path.join(config.data_dir, 'semnet', 'triples.npz')) as npz:
      subj = npz['subj']
      rel = npz['rel']
      obj = npz['obj']
    nsubj, nobj = sampler.sample(subj, rel, obj)
    feed_dict = {model.smoothing_placeholders['sn_relations']: rel,
                 model.smoothing_placeholders['sn_pos_subj']: subj,
                 model.smoothing_placeholders['sn_pos_obj']: obj,
                 model.smoothing_placeholders['sn_neg_subj']: nsubj,
                 model.smoothing_placeholders['sn_neg_obj']: nobj}
    pscores, nscores = session.run([model.sn_pos_energy, model.sn_neg_energy], feed_dict)
    predictions = svm.predict(pscores.reshape(-1, 1)).tolist()
    labels = np.ones_like(pscores, dtype=np.int).tolist()
    predictions += svm.predict(nscores.reshape(-1, 1)).tolist()
    labels += np.zeros_like(nscores, dtype=np.int).tolist()
    print('SN Relation Triple Classification Accuracy:  %.4f' % metrics.accuracy_score(labels, predictions))
    print('SN Relation Triple Classification Precision: %.4f' % metrics.precision_score(labels, predictions))
    print('SN Relation Triple Classification Recall:    %.4f' % metrics.recall_score(labels, predictions))
    print(metrics.classification_report(labels, predictions))



if __name__ == "__main__":
  if config.eval_mode == 'sn':
    evaluate_sn()
  else:
    evaluate()
