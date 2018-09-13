import numpy as np
import sys
import json
import os
import csv
from tqdm import tqdm
from collections import defaultdict

csv.field_size_limit(sys.maxsize)


def knn(embeddings, idx, k):
  k = k+1
  e = np.expand_dims(embeddings[idx], axis=0)
  distances = np.linalg.norm(e - embeddings, axis=1)
  print(distances.shape)
  idxs = list(range(embeddings.shape[0]))
  idxs.sort(key=lambda i_: distances[i_])

  return zip(idxs[:k], [distances[i] for i in idxs[:k]])


def main():
  cui2names = defaultdict(list)
  with open('/home/rmm120030/working/umls-mke/umls/META/MRCONSO.RRF', 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in tqdm(reader, desc="reading mrconso", total=8157818):
      cui2names[row[0]].append(row[14])

  cui2id = json.load(open(os.path.join('/home/rmm120030/working/umls-mke/data', 'name2id.json')))
  id2cui = {v: k for k, v in cui2id.iteritems()}
  with np.load(sys.argv[1]) as npz:
    embeddings = npz['embs']
  print('Loaded embedding matrix: %s' % str(embeddings.shape))

  while True:
    cui = raw_input('enter CUI (or \'exit\' to stop): ')
    if cui == "exit":
      exit()
    if cui in cui2id:
      neihbors = knn(embeddings, cui2id[cui], 10)
      for i, dist in neihbors:
        c = id2cui[i]
        print(' %.6f - %s - %s' % (dist, c, cui2names[c][:5]))
    else:
      print('No embedding for CUI %s' % cui)


if __name__ == "__main__":
  main()
