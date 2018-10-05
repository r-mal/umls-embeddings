import argparse
import csv
import json
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from create_test_set import split


def process_mapping(umls_dir, data_dir):
  print('Creating mapping of semtypes to cuis and vice versa...')
  name2id = json.load(open(os.path.join(data_dir, 'name2id.json')))
  cui2semtypes = defaultdict(list)
  semtype2cuis = defaultdict(list)
  with open(os.path.join(umls_dir, 'META', 'MRSTY.RRF'), 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in tqdm(reader, desc="reading", total=3395307):
      cui = row[0].strip()
      if cui in name2id:
        cid = name2id[cui]
        tui = row[1]
        if tui in name2id:
          tid = name2id[tui]
        else:
          tid = len(name2id)
          name2id[tui] = tid
        cui2semtypes[cid].append(tid)
        semtype2cuis[tid].append(cid)
  print('Processed type mappings for %d semantic types' % len(semtype2cuis))
  c2s_lens = sorted([len(l) for l in cui2semtypes.values()])
  print('Maximum # semtypes for a cui: %d' % max(c2s_lens))
  print('Average # semtypes for a cui: %.2f' % (float(sum(c2s_lens)) / len(c2s_lens)))
  print('Median  # semtypes for a cui: %.2f' % c2s_lens[len(c2s_lens)/2])

  s2c_lens = sorted([len(l) for l in semtype2cuis.values()])
  print('Maximum # cuis for a semtype: %d' % max(s2c_lens))
  print('Average # cuis for a semtype: %.2f' % (float(sum(s2c_lens)) / len(s2c_lens)))
  print('Median  # cuis for a semtype: %.2f' % s2c_lens[len(s2c_lens)/2])
  print('%% under 1k: %.4f' % (float(len([l for l in s2c_lens if l < 1000])) / len(s2c_lens)))
  print('%% under 2k: %.4f' % (float(len([l for l in s2c_lens if l < 2000])) / len(s2c_lens)))

  # json.dump(name2id, open('/home/rmm120030/working/umls-mke/data/name2id.json', 'w+'))
  json.dump(cui2semtypes, open(os.path.join(data_dir, 'semnet', 'cui2semtpyes.json'), 'w+'))
  json.dump(semtype2cuis, open(os.path.join(data_dir, 'semnet', 'semtype2cuis.json'), 'w+'))


def semnet_triples(umls_dir, data_dir):
  print('Creating semantic network triples...')
  name2id = json.load(open(os.path.join(data_dir, 'name2id.json')))

  total_relations = 0
  new_relations = 0
  tui2id = {}
  # relations which have a metathesaurus analog are mapped to the MT embedding
  with open(os.path.join(umls_dir, 'NET', 'SRDEF')) as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
      tui = row[1]
      if row[0] == 'RL':
        total_relations += 1
        rel = row[2]
        if rel in name2id:
          print('reusing relation embedding for %s' % rel)
          name2id[tui] = name2id[rel]
        elif tui not in name2id:
          new_relations += 1
          name2id[tui] = len(name2id)
      else:
        if tui not in name2id:
          name2id[tui] = len(name2id)
      tui2id[tui] = name2id[tui]

  print('Created %d of %d new relations' % (new_relations, total_relations))
  print('%d total embeddings' % len(name2id))
  json.dump(name2id, open(os.path.join(data_dir, 'name2id.json'), 'w+'))
  json.dump(tui2id, open(os.path.join(data_dir, 'semnet', 'tui2id.json'), 'w+'))

  subj, rel, obj = [], [], []
  with open(os.path.join(umls_dir, 'NET', 'SRSTRE1'), 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
      subj.append(name2id[row[0]])
      rel.append(name2id[row[1]])
      obj.append(name2id[row[2]])

  print('Saving the %d triples of the semantic network graph' % len(rel))
  split(np.asarray(subj, dtype=np.int32),
        np.asarray(rel, dtype=np.int32),
        np.asarray(obj, dtype=np.int32),
        os.path.join(data_dir, 'semnet'),
        'semnet',
        600)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Extract relation triples into a compressed numpy file from MRCONSO.RRF')
  parser.add_argument('umls_dir', help='UMLS directory')
  parser.add_argument('--output', default='data', help='the compressed numpy file to be created')

  args = parser.parse_args()
  semnet_triples(args.umls_dir, args.output)
  process_mapping(args.umls_dir, args.output)
