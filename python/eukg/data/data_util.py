import numpy as np
from collections import defaultdict
from tqdm import tqdm
import sys
import math
import os
import json
import random


def negative_sampling_from_file(np_file):
  npz = np.load(np_file)
  subj, rel, obj = npz['subj'], npz['rel'], npz['obj']

  sampler = NegativeSampler(subj, rel, obj)
  neg_subj, neg_obj = sampler.sample(subj, rel, obj)

  npz.close()
  np.savez_compressed(np_file,
                      subj=subj,
                      rel=rel,
                      obj=obj,
                      nsubj=neg_subj,
                      nobj=neg_obj)


def load_metathesaurus_data(data_dir, val_proportion):
  random.seed(1337)

  cui2id = json.load(open(os.path.join(data_dir, 'name2id.json')))
  npz = np.load(os.path.join(data_dir, 'metathesaurus', 'train.npz'))
  data = dict(npz.iteritems())
  npz.close()

  perm = np.random.permutation(np.arange(len(data['rel'])))
  num_val = int(math.ceil(len(perm) * val_proportion))
  val_idx = perm[:num_val]
  train_idx = perm[num_val:]

  return cui2id, data, train_idx, val_idx


def load_semantic_network_data(data_dir, data_map):
  type2cuis = json.load(open(os.path.join(data_dir, 'semnet', 'semtype2cuis.json')))
  npz = np.load(os.path.join(data_dir, 'semnet', 'triples.npz'))
  for key, val in npz.iteritems():
    data_map['sn_' + key] = val
  npz.close()

  return type2cuis


def load_metathesaurus_test_data(data_dir):
  npz = np.load(os.path.join(data_dir, 'metathesaurus', 'test.npz'))
  data = dict(npz.iteritems())
  npz.close()

  return data


def save_config(outdir, config):
  print('saving config to %s' % outdir)
  with open('%s/config.json' % outdir, 'w+') as f:
    json.dump(config.flag_values_dict(), f)


def main():
  negative_sampling_from_file(sys.argv[1])


class NegativeSampler:
  def __init__(self, subj, rel, obj, name, cachedir="/tmp"):
    cachedir = os.path.join(cachedir, name)
    # if os.path.exists(cachedir):
    #   print('loading negative sampler maps from %s' % cachedir)
    #   self.sr2o = defaultdict(list, json.load(open(os.path.join(cachedir, 'sr2o.json'))))
    #   self.or2s = defaultdict(list, json.load(open(os.path.join(cachedir, 'or2s.json'))))
    #   self.concepts = json.load(open(os.path.join(cachedir, 'concepts.json')))
    # else:
    self.sr2o = defaultdict(set)
    self.or2s = defaultdict(set)
    concepts = set()
    print("\n")
    for s, r, o in tqdm(zip(subj, rel, obj), desc='building triple maps', total=len(subj)):
      self.sr2o[(s, r)].add(o)
      self.or2s[(o, r)].add(s)
      concepts.update([s, o])
    self.concepts = list(concepts)

    # os.makedirs(cachedir)
    # json.dump({k: list(v) for k, v in self.sr2o.iteritems()}, open(os.path.join(cachedir, 'sr2o.json'), 'w+'))
    # json.dump({k: list(v) for k, v in self.or2s.iteritems()}, open(os.path.join(cachedir, 'or2s.json'), 'w+'))
    # json.dump(self.concepts, open(os.path.join(cachedir, 'concepts.json'), 'w+'))

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
      ns, no = self._neg_sample(s, r, o, random.random > 0.5)
      neg_subj.append(ns)
      neg_obj.append(no)

    return np.asarray(neg_subj, dtype=np.int32), np.asarray(neg_obj, dtype=np.int32)

  def _sample_k(self, subj, rel, obj, k):
    neg_subj = []
    neg_obj = []
    for i in xrange(k):
      ns, no = self._neg_sample(subj, rel, obj, random.random > 0.5)
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

    return np.asarray(subj_samples), np.asarray(obj_samples)


if __name__ == "__main__":
  main()
