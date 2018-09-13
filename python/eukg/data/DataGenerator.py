import numpy as np
import random
import math
from itertools import izip
from collections import defaultdict
from tqdm import tqdm
import os
import ujson

import data_util


class DataGenerator:
  def __init__(self, data, train_idx, val_idx, config, type2cuis=None, test_mode=False):
    self.data = data
    self.train_idx = train_idx
    self.val_idx = val_idx
    self.config = config
    self._sampler = None
    self._sn_sampler = None
    if not config.no_semantic_network:
      assert type2cuis
      self._sn_sampler = NegativeSampler(data['sn_subj'], data['sn_rel'], data['sn_obj'], 'semnet')
      # if we wish to train this model for sn eval, only use
      if config.sn_eval:
        with np.load(os.path.join(config.data_dir, 'semnet', 'train.npz')) as sn_npz:
          for key, val in sn_npz.iteritems():
            self.data['sn_' + key] = val

    self.type2cuis = type2cuis
    self.test_mode = test_mode

  @property
  def sn_sampler(self):
    # if self._sn_sampler is None:
    #   self._sn_sampler = self.init_sn_sampler()
    return self._sn_sampler

  @property
  def sampler(self):
    if self._sampler is None:
      self._sampler = self.init_sampler()
    return self._sampler

  # must include test data in negative sampler
  def init_sampler(self):
    if self.test_mode:
      _, test_data, _, _ = data_util.load_metathesaurus_data(self.config.data_dir, 0.)
    else:
      test_data = data_util.load_metathesaurus_test_data(self.config.data_dir)
    valid_triples = set()
    for s, r, o in zip(self.data['subj'], self.data['rel'], self.data['obj']):
      valid_triples.add((s, r, o))
    for s, r, o in zip(test_data['subj'], test_data['rel'], test_data['obj']):
      valid_triples.add((s, r, o))

    return NegativeSampler(valid_triples=valid_triples, name='mt')

  def generate_mt(self, is_training):
    idxs = self.train_idx if is_training else self.val_idx
    batch_size = self.config.batch_size
    subj, rel, obj = self.data['subj'], self.data['rel'], self.data['obj']
    nsubj, nobj = self.sampler.sample(subj, rel, obj)
    num_batches = int(math.floor(float(len(idxs)) / batch_size))
    print('\n\ngenerating %d batches' % num_batches)
    for b in xrange(num_batches):
      idx = idxs[b * batch_size: (b + 1) * batch_size]
      yield rel[idx], subj[idx], obj[idx], nsubj[idx], nobj[idx]

  def generate_mt_gen_mode(self, is_training):
    idxs = self.train_idx if is_training else self.val_idx
    batch_size = self.config.batch_size
    subj, rel, obj = self.data['subj'], self.data['rel'], self.data['obj']
    num_batches = int(math.floor(float(len(idxs)) / batch_size))
    print('\n\ngenerating %d batches in generation mode' % num_batches)
    for b in xrange(num_batches):
      idx = idxs[b * batch_size: (b + 1) * batch_size]
      sampl_subj, sampl_obj = self.sampler.sample_for_generator(subj[idx], rel[idx], obj[idx],
                                                                self.config.num_generator_samples)
      yield rel[idx], subj[idx], obj[idx], sampl_subj, sampl_obj

  def generate_sn(self, is_training):
    print('\n\ngenerating SN data')
    if is_training:
      idxs = np.random.permutation(np.arange(len(self.data['sn_rel'])))
      data = self.data
      subj, rel, obj = data['sn_subj'], data['sn_rel'], data['sn_obj']
      nsubj, nobj = self.sn_sampler.sample(subj, rel, obj)
      sn_offset = 0
      batch_size = self.config.batch_size / 4
      while True:
        idx, sn_offset = get_next_k_idxs(idxs, batch_size, sn_offset)
        types = np.unique([subj[idx], obj[idx], nsubj[idx], nobj[idx]])
        concepts = np.zeros([len(types), self.config.max_concepts_per_type], dtype=np.int32)
        concept_lens = np.zeros([len(types)], dtype=np.int32)
        for i, tid in enumerate(types):
          concepts_of_type_t = self.type2cuis[tid] if tid in self.type2cuis else []
          random.shuffle(concepts_of_type_t)
          concepts_of_type_t = concepts_of_type_t[:self.config.max_concepts_per_type]
          concept_lens[i] = len(concepts_of_type_t)
          concepts[i, :len(concepts_of_type_t)] = concepts_of_type_t

        yield rel[idx], subj[idx], obj[idx], nsubj[idx], nobj[idx], \
              concepts, concept_lens, types
    else:
      while True:
        yield [0], [0], [0], [0], [0], np.zeros([1, 1000], dtype=np.int32), [1], [0]

  def generate_sn_gen_mode(self, is_training):
    print('\n\ngenerating SN data in generation mode')
    num_samples = 10
    if is_training:
      idxs = np.random.permutation(np.arange(len(self.data['sn_rel'])))
      subj, rel, obj = self.data['sn_subj'], self.data['sn_rel'], self.data['sn_obj']
      sn_offset = 0
      batch_size = self.config.batch_size / 4
      while True:
        idx, sn_offset = get_next_k_idxs(idxs, batch_size, sn_offset)
        subj_ = subj[idx]
        rel_ = rel[idx]
        obj_ = obj[idx]
        subj_samples, obj_samples = self.sn_sampler.sample_for_generator(subj_, rel_, obj_, num_samples)

        types = np.unique(np.concatenate([subj_, obj_, subj_samples.flatten(), obj_samples.flatten()]))
        concepts = np.zeros([len(types), self.config.max_concepts_per_type], dtype=np.int32)
        concept_lens = np.zeros([len(types)], dtype=np.int32)
        for i, tid in enumerate(types):
          concepts_of_type_t = self.type2cuis[tid] if tid in self.type2cuis else []
          random.shuffle(concepts_of_type_t)
          concepts_of_type_t = concepts_of_type_t[:self.config.max_concepts_per_type]
          concept_lens[i] = len(concepts_of_type_t)
          concepts[i, :len(concepts_of_type_t)] = concepts_of_type_t

        yield rel_, subj_, obj_, subj_samples, obj_samples, \
              concepts, concept_lens, types
    else:
      while True:
        yield [0], [0], [0], [0, 0], [0, 0], np.zeros([1, 1000], dtype=np.int32), [1], [0]

  def num_train_batches(self):
    return int(math.floor(float(len(self.train_idx)) / self.config.batch_size))

  def num_val_batches(self):
    return int(math.floor(float(len(self.val_idx)) / self.config.batch_size))


class NegativeSampler:
  def __init__(self, subj=None, rel=None, obj=None, name=None, cachedir="/home/rmm120030/working/umls-mke/.cache",
               valid_triples=None):
    # cachedir = os.path.join(cachedir, name)
    # if os.path.exists(cachedir):
    #   start = time.time()
    #   print('loading negative sampler maps from %s' % cachedir)
    #   self.sr2o = load_dict(os.path.join(cachedir, 'sr2o'))
    #   self.or2s = load_dict(os.path.join(cachedir, 'or2s'))
    #   self.concepts = ujson.load(open(os.path.join(cachedir, 'concepts.json')))
    #   print('done! Took %.2f seconds' % (time.time() - start))
    # else:
    self.sr2o = defaultdict(set)
    self.or2s = defaultdict(set)
    concepts = set()
    triples = zip(subj, rel, obj) if valid_triples is None else valid_triples
    for s, r, o in tqdm(triples, desc='building triple maps', total=len(triples)):
      # s, r, o = int(s), int(r), int(o)
      self.sr2o[(s, r)].add(o)
      self.or2s[(o, r)].add(s)
      concepts.update([s, o])
    self.concepts = list(concepts)

      # print('\n\ncaching negative sampler maps to %s' % cachedir)
      # os.makedirs(cachedir)
      # save_dict(self.sr2o, os.path.join(cachedir, 'sr2o'))
      # save_dict(self.or2s, os.path.join(cachedir, 'or2s'))
      # ujson.dump(self.concepts, open(os.path.join(cachedir, 'concepts.json'), 'w+'))
      # print('done!')

  def _neg_sample(self, s_, r_, o_, replace_s):
    while True:
      c = random.choice(self.concepts)
      if replace_s and c not in self.or2s[(o_, r_)]:
        return c, o_
      elif not replace_s and c not in self.sr2o[(s_, r_)]:
        return s_, c

  def sample(self, subj, rel, obj):
    neg_subj = []
    neg_obj = []
    print("\n")
    for s, r, o in tqdm(zip(subj, rel, obj), desc='negative sampling', total=len(subj)):
      ns, no = self._neg_sample(s, r, o, random.random() > 0.5)
      neg_subj.append(ns)
      neg_obj.append(no)

    return np.asarray(neg_subj, dtype=np.int32), np.asarray(neg_obj, dtype=np.int32)

  def _sample_k(self, subj, rel, obj, k):
    neg_subj = []
    neg_obj = []
    for i in xrange(k):
      ns, no = self._neg_sample(subj, rel, obj, random.random() > 0.5)
      neg_subj.append(ns)
      neg_obj.append(no)

    return neg_subj, neg_obj

  def sample_for_generator(self, subj_array, rel_array, obj_array, k):
    subj_samples = []
    obj_samples = []
    for s, r, o in zip(subj_array, rel_array, obj_array):
      ns, no = self._sample_k(s, r, o, k)
      subj_samples.append(ns)
      obj_samples.append(no)

    return np.asarray(subj_samples, dtype=np.int32), np.asarray(obj_samples, dtype=np.int32)

  def invalid_concepts(self, subj, rel, obj, replace_subj):
    if replace_subj:
      return [c for c in self.concepts if c not in self.or2s[(obj, rel)]]
    else:
      return [c for c in self.concepts if c not in self.sr2o[(subj, rel)]]


def get_next_k_idxs(all_idxs, k, offset):
  if offset + k < len(all_idxs):
    idx = all_idxs[offset: offset + k]
    offset += k
  else:
    random.shuffle(all_idxs)
    offset = k
    idx = all_idxs[:offset]
  return idx, offset


def wrap_generators(mt_gen, sn_gen, is_training):
  if is_training:
    for mt_batch, sn_batch in izip(mt_gen(True), sn_gen(True)):
      yield mt_batch + sn_batch
  else:
    for b in mt_gen(True):
      yield b


def save_dict(d, savepath):
  keys, values = [], []
  for k, v in d.iteritems():
    keys.append(list(k))
    values.append(list(v))

  ujson.dump(keys, open(savepath + '_keys.json', 'w+'))
  ujson.dump(values, open(savepath + '_values.json', 'w+'))


def load_dict(savepath):
  keys = ujson.load(open(savepath + '_keys.json'))
  values = ujson.load(open(savepath + '_values.json'))
  return {tuple(k): set(v) for k, v in izip(keys, values)}
