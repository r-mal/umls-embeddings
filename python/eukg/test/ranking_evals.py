import tensorflow as tf
import numpy as np
import random
import math
import os
import json
from tqdm import tqdm
from collections import defaultdict

from ..data import data_util, DataGenerator
from .. import Config, train
from ..threading_util import synchronized, parallel_stream


config = Config.flags


def save_ranks():
  random.seed(1337)
  config.batch_size = 4096

  cui2id, train_data, _, _ = data_util.load_metathesaurus_data(config.data_dir, config.val_proportion)
  id2cui = {v: k for k, v in cui2id.iteritems()}
  test_data = data_util.load_metathesaurus_test_data(config.data_dir)
  print('Loaded %d test triples from %s' % (len(test_data['rel']), config.data_dir))

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

    if not config.save_ranks:
      print('WARNING: ranks will not be saved! This run should only be for debugging purposes!')

    outdir = os.path.join(config.eval_dir, config.run_name)
    if not os.path.exists(outdir):
      os.makedirs(outdir)

    chunk_size = (1. / config.num_shards) * len(test_data['rel'])
    print('Chunk size: %f' % chunk_size)
    start = int((config.shard - 1) * chunk_size)
    end = (len(test_data['rel']) if config.shard == int(config.num_shards) else int(config.shard * chunk_size)) \
          if config.save_ranks else start + 1000
    print('Processing data from idx %d to %d' % (start, end))

    sampler = DataGenerator.NegativeSampler(valid_triples=valid_triples, name='gan')
    global ranks
    global first
    ranks = []
    first = True
    open_file = lambda: open(os.path.join(outdir, 'ranks_%d.json' % config.shard), 'w+') \
      if config.save_ranks else open("/dev/null")
    with open_file() as f, tqdm(total=(end-start)) as pbar:
      if config.save_ranks:
        f.write('[')

      # define thread function
      def calculate(triple):
        s, r, o = triple
        # subj
        invalid_concepts = [s] + synchronized(lambda: sampler.invalid_concepts(s, r, o, True))
        subj_scores, subj_ranking = calculate_scores(s, r, o, True, invalid_concepts, session, model, config.batch_size)
        srank = subj_ranking[s] if s in subj_ranking else -1

        # obj
        invalid_concepts = [o] + synchronized(lambda: sampler.invalid_concepts(s, r, o, False))
        obj_scores, obj_ranking = calculate_scores(s, r, o, False, invalid_concepts, session, model, config.batch_size)
        orank = obj_ranking[o] if o in obj_ranking else -1
        json_string = json.dumps([str(srank),
                                  str(orank),
                                  str(obj_scores[o]),
                                  id2cui[s], id2cui[r], id2cui[o]])
        return srank, orank, json_string

      # define save function
      def save(future):
        global ranks
        global first

        srank, orank, json_string = future.result()
        ranks += [srank, orank]
        if config.save_ranks:
          if not first:
            f.write(',')
          first = False
          f.write(json_string)
        npr = np.asarray(ranks, dtype=np.float)
        if config.save_ranks:
          pbar.set_description('Srank: %5d. Orank: %5d.' % (srank, orank))
        else:
          pbar.set_description('Srank: %5d. Orank: %5d. MRR: %.4f. H@10: %.4f' %
                               (srank, orank, mrr(npr), hits_at_10(npr)))
        pbar.update()

      parallel_stream(zip(test_data['subj'][start:end], test_data['rel'][start:end], test_data['obj'][start:end]),
                      parallelizable_fn=calculate,
                      future_consumer=save)

      # finish json
      if config.save_ranks:
        f.write(']')

  ranks_np = np.asarray(ranks, dtype=np.float)
  print('Mean Reciprocal Rank of shard %d: %.4f' % (config.shard, mrr(ranks_np)))
  print('Mean Rank of shard %d: %.2f' % (config.shard, mr(ranks_np)))
  print('Hits @ 10 of shard %d: %.4f' % (config.shard, hits_at_10(ranks_np)))


def calculate_scores(subj, rel, obj, replace_subject, concept_ids, session, model, batch_size):
  num_batches = int(math.ceil(float(len(concept_ids))/batch_size))
  subjects = np.full(batch_size, subj, dtype=np.int32)
  relations = np.full(batch_size, rel, dtype=np.int32)
  objects = np.full(batch_size, obj, dtype=np.int32)
  scores = {}

  for b in xrange(num_batches):
    concepts = concept_ids[b*batch_size:(b+1)*batch_size]
    feed_dict = {model.pos_subj: subjects,
                 model.relations: relations,
                 model.pos_obj: objects}

    # pad concepts if necessary
    if len(concepts) < batch_size:
      concepts = np.pad(concepts, (0, batch_size - len(concepts)), mode='constant', constant_values=0)

    # replace subj/obj in feed dict
    if replace_subject:
      feed_dict[model.pos_subj] = concepts
    else:
      feed_dict[model.pos_obj] = concepts

    # calculate energies
    energies = session.run(model.pos_energy, feed_dict)

    # store scores
    for i, cid in enumerate(concept_ids[b*batch_size:(b+1)*batch_size]):
      scores[cid] = energies[i]

  ranking = sorted(scores.keys(), key=lambda k: scores[k])

  rank_map = {}
  prev_rank = 0
  prev_score = -1
  total = 1
  for c in ranking:
    # if c has a lower score than prev
    if scores[c] > prev_score:
      # increment the rank
      prev_rank = total
      # update score
      prev_score = scores[c]
    total += 1
    rank_map[c] = prev_rank

  return scores, rank_map


def save_ranks_sn():
  random.seed(1337)
  config.batch_size = 4096
  assert not config.no_semantic_network

  data = {}
  cui2id, _, _, _ = data_util.load_metathesaurus_data(config.data_dir, 0.)
  id2cui = {v: k for k, v in cui2id.iteritems()}
  _ = data_util.load_semantic_network_data(config.data_dir, data)
  subj, rel, obj = data['sn_subj'], data['sn_rel'], data['sn_obj']
  n = len(rel)
  print('Loaded %d sn triples from %s' % (n, config.data_dir))

  valid_triples = set()
  for trip in zip(subj, rel, obj):
    valid_triples.add(trip)
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

    if not config.save_ranks:
      print('WARNING: ranks will not be saved! This run should only be for debugging purposes!')

    outdir = os.path.join(config.eval_dir, config.run_name)
    if not os.path.exists(outdir):
      os.makedirs(outdir)

    sampler = DataGenerator.NegativeSampler(valid_triples=valid_triples, name='gan')
    global ranks
    global first
    ranks = []
    first = True
    open_file = lambda: open(os.path.join(outdir, 'sn_ranks.json'), 'w+') \
      if config.save_ranks else open("/dev/null")
    with open_file() as f, tqdm(total=600) as pbar:
      if config.save_ranks:
        f.write('[')

      # define thread function
      def calculate(triple):
        s, r, o = triple
        # subj
        invalid_concepts = [s] + synchronized(lambda: sampler.invalid_concepts(s, r, o, True))
        subj_scores, subj_ranking = calculate_scores_sn(s, r, o, True, invalid_concepts, session, model, config.batch_size)
        srank = subj_ranking[s] if s in subj_ranking else -1

        # obj
        invalid_concepts = [o] + synchronized(lambda: sampler.invalid_concepts(s, r, o, False))
        obj_scores, obj_ranking = calculate_scores_sn(s, r, o, False, invalid_concepts, session, model, config.batch_size)
        orank = obj_ranking[o] if o in obj_ranking else -1
        json_string = json.dumps([str(srank),
                                  str(orank),
                                  str(obj_scores[o]),
                                  id2cui[s], id2cui[r], id2cui[o]])
        return srank, orank, json_string

      # define save function
      def save(future):
        global ranks
        global first

        srank, orank, json_string = future.result()
        ranks += [srank, orank]
        if config.save_ranks:
          if not first:
            f.write(',')
          first = False
          f.write(json_string)
        npr = np.asarray(ranks, dtype=np.float)
        pbar.set_description('Srank: %5d. Orank: %5d. MRR: %.4f. H@10: %.4f' %
                             (srank, orank, mrr(npr), hits_at_10(npr)))
        pbar.update()

      with np.load(os.path.join(config.data_dir, 'semnet', 'test.npz')) as sn_npz:
        subj, rel, obj = sn_npz['subj'], sn_npz['rel'], sn_npz['obj']
      parallel_stream(zip(subj, rel, obj),
                      parallelizable_fn=calculate,
                      future_consumer=save)

      # finish json
      if config.save_ranks:
        f.write(']')

  ranks_np = np.asarray(ranks, dtype=np.float)
  print('Mean Reciprocal Rank: %.4f' % (mrr(ranks_np)))
  print('Mean Rank: %.2f' % (mr(ranks_np)))
  print('Hits @ 10: %.4f' % (hits_at_10(ranks_np)))


def calculate_scores_sn(subj, rel, obj, replace_subject, concept_ids, session, model, batch_size):
  num_batches = int(math.ceil(float(len(concept_ids))/batch_size))
  subjects = np.full(batch_size, subj, dtype=np.int32)
  relations = np.full(batch_size, rel, dtype=np.int32)
  objects = np.full(batch_size, obj, dtype=np.int32)
  scores = {}

  for b in xrange(num_batches):
    concepts = concept_ids[b*batch_size:(b+1)*batch_size]
    feed_dict = {model.smoothing_placeholders['sn_pos_subj']: subjects,
                 model.smoothing_placeholders['sn_relations']: relations,
                 model.smoothing_placeholders['sn_pos_obj']: objects}

    # pad concepts if necessary
    if len(concepts) < batch_size:
      concepts = np.pad(concepts, (0, batch_size - len(concepts)), mode='constant', constant_values=0)

    # replace subj/obj in feed dict
    if replace_subject:
      feed_dict[model.smoothing_placeholders['sn_pos_subj']] = concepts
    else:
      feed_dict[model.smoothing_placeholders['sn_pos_obj']] = concepts

    # calculate energies
    energies = session.run(model.sn_pos_energy, feed_dict)

    # store scores
    for i, cid in enumerate(concept_ids[b*batch_size:(b+1)*batch_size]):
      scores[cid] = energies[i]

  ranking = sorted(scores.keys(), key=lambda k: scores[k])

  rank_map = {}
  prev_rank = 0
  prev_score = -1
  total = 1
  for c in ranking:
    # if c has a lower score than prev
    if scores[c] > prev_score:
      # increment the rank
      prev_rank = total
      # update score
      prev_score = scores[c]
    total += 1
    rank_map[c] = prev_rank

  return scores, rank_map


def mrr(ranks_np):
  return float(np.mean(1. / ranks_np))


def mr(ranks_np):
  return float(np.mean(ranks_np))


def hits_at_10(ranks_np):
  return float(len(ranks_np[ranks_np <= 10])) / len(ranks_np)


def calculate_ranking_evals():
  outdir = os.path.join(config.eval_dir, config.run_name)

  ppa = float(str(next(open(os.path.join(outdir, 'ppa.txt')))).strip())
  print('PPA:  %.4f' % ppa)

  ranks = []
  for fname in os.listdir(outdir):
    full_path = os.path.join(outdir, fname)
    if os.path.isfile(full_path) and fname.startswith('ranks_'):
      for fields in json.load(open(full_path)):
        ranks += [fields[0], fields[1]]
  ranks_np = np.asarray(ranks, dtype=np.float)
  mrr_ = mrr(ranks_np)
  mr_ = mr(ranks_np)
  hat10 = hits_at_10(ranks_np)
  print('MRR:  %.4f' % mrr_)
  print('MR:   %.2f' % mr_)
  print('H@10: %.4f' % hat10)

  with open(os.path.join(outdir, 'ranking_evals.tsv'), 'w+') as f:
    f.write('ppa\t%f\n' % ppa)
    f.write('mrr\t%f\n' % mrr_)
    f.write('mr\t%f\n' % mr_)
    f.write('h@10\t%f' % hat10)


def calculate_ranking_evals_per_rel():
  outdir = os.path.join(config.eval_dir, config.run_name)

  ranks = defaultdict(list)
  for fname in os.listdir(outdir):
    full_path = os.path.join(outdir, fname)
    if os.path.isfile(full_path) and fname.startswith('ranks_'):
      for fields in json.load(open(full_path)):
        ranks[fields[4]] += [fields[0], fields[1]]
  print('Gathered %d rankings' % len(ranks))

  for rel, rl in ranks.iteritems():
    ranks_np = np.asarray(rl, dtype=np.float)
    ranks[rel] = [mrr(ranks_np), mr(ranks_np), hits_at_10(ranks_np), len(rl)]

  relations = ranks.keys()
  relations.sort(key=lambda x: ranks[x][1])

  with open(os.path.join(outdir, 'ranking_evals_per_rel.tsv'), 'w+') as f:
    for rel in relations:
      [mrr_, mr_, hat10, count] = ranks[rel]
      print('----------%s(%d)----------' % (rel, count))
      print('MRR:  %.4f' % mrr_)
      print('MR:   %.2f' % mr_)
      print('H@10: %.4f' % hat10)

      f.write('%s (%d)\n' % (rel, count))
      f.write('mrr\t%f\n' % mrr_)
      f.write('mr\t%f\n' % mr_)
      f.write('h@10\t%f\n\n' % hat10)


def split_ranking_evals():
  outdir = os.path.join(config.eval_dir, config.run_name)

  ppa = float(str(next(open(os.path.join(outdir, 'ppa.txt')))).strip())
  print('PPA:  %.4f' % ppa)

  s_ranks = []
  o_ranks = []
  for fname in os.listdir(outdir):
    full_path = os.path.join(outdir, fname)
    if os.path.isfile(full_path) and fname.startswith('ranks_'):
      for fields in json.load(open(full_path)):
        s_ranks += [fields[0]]
        o_ranks += [fields[1]]

  def report(ranks):
    ranks_np = np.asarray(ranks, dtype=np.float)
    mrr_ = mrr(ranks_np)
    mr_ = mr(ranks_np)
    hat10 = hits_at_10(ranks_np)
    print('MRR:  %.4f' % mrr_)
    print('MR:   %.2f' % mr_)
    print('H@10: %.4f' % hat10)
  report(s_ranks)
  report(o_ranks)


def fix_json():
  config = Config.flags
  outdir = os.path.join(config.eval_dir, config.run_name)

  ppa = float(str(next(open(os.path.join(outdir, 'ppa.txt')))).strip())
  print('PPA:  %.4f' % ppa)

  ranks = []
  for fname in os.listdir(outdir):
    full_path = os.path.join(outdir, fname)
    if os.path.isfile(full_path) and fname.startswith('ranks_'):
      json_string = '[' + str(next(open(full_path))).strip()
      with open(full_path, 'w+') as f:
        f.write(json_string)
      for fields in json.load(open(full_path)):
        ranks += [fields[0], fields[2]]
  ranks_np = np.asarray(ranks, dtype=np.float)
  mrr_ = mrr(ranks_np)
  mr_ = mr(ranks_np)
  hat10 = hits_at_10(ranks_np)
  print('MRR:  %.4f' % mrr_)
  print('MR:   %.2f' % mr_)
  print('H@10: %.4f' % hat10)

  with open(os.path.join(outdir, 'ranking_evals.tsv'), 'w+') as f:
    f.write('ppa\t%f\n' % ppa)
    f.write('mrr\t%f\n' % mrr_)
    f.write('mr\t%f\n' % mr_)
    f.write('h@10\t%f' % hat10)


if __name__ == "__main__":
  if config.eval_mode == "save":
    save_ranks()
  elif config.eval_mode == "save-sn":
    save_ranks_sn()
  elif config.eval_mode == "calc":
    calculate_ranking_evals()
  elif config.eval_mode == "calc-rel":
    calculate_ranking_evals_per_rel()
  else:
    raise Exception('Unrecognized eval_mode: %s' % config.eval_mode)
