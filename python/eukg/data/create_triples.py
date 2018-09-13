import numpy as np
import sys
import os
import csv
import json
from tqdm import tqdm


def metathesaurus_triples(rrf_file, output_dir, valid_relations):
  triples = set()
  conc2id = {}

  def add_concept(conc):
    if conc in conc2id:
      cid = conc2id[conc]
    else:
      cid = len(conc2id)
      conc2id[conc] = cid
    return cid

  with open(rrf_file, 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in tqdm(reader, desc="reading", total=37207861):
      if row[7] in valid_relations:
        sid = add_concept(row[0])
        rid = add_concept(row[7])
        oid = add_concept(row[4])
        triples.add((sid, rid, oid))

  subjs, rels, objs = zip(*triples)
  snp = np.asarray(subjs, dtype=np.int32)
  rnp = np.asarray(rels, dtype=np.int32)
  onp = np.asarray(objs, dtype=np.int32)

  id2conc = {v: k for k, v in conc2id.iteritems()}
  concepts = [id2conc[i] for i in np.unique(np.concatenate((subjs, objs)))]
  relations = [id2conc[i] for i in set(rels)]

  print("Saving %d unique triples to %s. %d concepts spanning %d relations" % (rnp.shape[0], output_dir, len(concepts),
                                                                               len(relations)))

  np.savez_compressed(os.path.join(output_dir, 'triples'),
                      subj=snp,
                      rel=rnp,
                      obj=onp)
  json.dump(conc2id, open(os.path.join(output_dir, 'name2id.json'), 'w+'))
  json.dump(concepts, open(os.path.join(output_dir, 'concept_vocab.json'), 'w+'))
  json.dump(relations, open(os.path.join(output_dir, 'relation_vocab.json'), 'w+'))


def metathesaurus_triples_trimmed(rrf_file, output_dir, valid_concepts, valid_relations, important_concepts=None):
  triples = set()
  conc2id = {}

  def add_concept(conc):
    if conc in conc2id:
      cid = conc2id[conc]
    else:
      cid = len(conc2id)
      conc2id[conc] = cid
    return cid

  with open(rrf_file, 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in tqdm(reader, desc="reading", total=37207861):
      if (row[0] in valid_concepts and row[4] in valid_concepts and row[7] in valid_relations) or \
           (important_concepts is not None and row[7] != '' and (row[0] in important_concepts or row[4] in important_concepts)):
        sid = add_concept(row[0])
        rid = add_concept(row[7])
        oid = add_concept(row[4])
        triples.add((sid, rid, oid))

  subjs, rels, objs = zip(*triples)
  snp = np.asarray(subjs, dtype=np.int32)
  rnp = np.asarray(rels, dtype=np.int32)
  onp = np.asarray(objs, dtype=np.int32)

  id2conc = {v: k for k, v in conc2id.iteritems()}
  concepts = [id2conc[i] for i in np.unique(np.concatenate((subjs, objs)))]
  relations = [id2conc[i] for i in rels]

  print("Saving %d unique triples to %s" % (rnp.shape[0], output_dir))

  np.savez_compressed(os.path.join(output_dir, 'triples'),
                      subj=snp,
                      rel=rnp,
                      obj=onp)
  json.dump(conc2id, open(os.path.join(output_dir, 'name2id.json'), 'w+'))
  json.dump(concepts, open(os.path.join(concepts, 'concept_vocab.json'), 'w+'))
  json.dump(relations, open(os.path.join(relations, 'relation_vocab.json'), 'w+'))


def main():
  rrf_file = sys.argv[1]
  valid_dir = sys.argv[2]
  output = sys.argv[3]

  valid_relations = set([rel.strip() for rel in open(os.path.join(valid_dir, 'valid_rels.txt'))])

  metathesaurus_triples(rrf_file, output, valid_relations)


if __name__ == "__main__":
  main()
