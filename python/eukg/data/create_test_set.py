import numpy as np
import sys
import os


def split(subj, rel, obj, data_dir, graph_name='metathesaurus', num_test=100000):
  valid_triples = set()
  for s, r, o in zip(subj, rel, obj):
    valid_triples.add((s, r, o))
  subj = np.asarray([s for s, _, _ in valid_triples])
  rel = np.asarray([r for _, r, _ in valid_triples])
  obj = np.asarray([o for _, _, o in valid_triples])

  perm = np.random.permutation(np.arange(len(rel)))
  test_idx = perm[:num_test]
  train_idx = perm[num_test:]
  print('created train/test splits (%d/%d)' % (len(train_idx), len(test_idx)))

  np.savez_compressed(os.path.join(data_dir, graph_name, 'train.npz'),
                      subj=subj[train_idx],
                      rel=rel[train_idx],
                      obj=obj[train_idx])
  print('saved train set')

  np.savez_compressed(os.path.join(data_dir, graph_name, 'test.npz'),
                      subj=subj[test_idx],
                      rel=rel[test_idx],
                      obj=obj[test_idx])
  print('saved test set')


def from_train_file(data_dir, graph_name='metathesaurus', num_test=100000):
  npz = np.load(os.path.join(data_dir, graph_name, 'triples.npz'))
  data = dict(npz.iteritems())
  npz.close()
  print('read all data')
  split(data['subj'], data['rel'], data['obj'], data_dir, graph_name, num_test)


def from_train_test_files():
  data_dir = sys.argv[1]
  npz = np.load(os.path.join(data_dir, 'metathesaurus', 'train.npz'))
  data = dict(npz.iteritems())
  npz.close()
  npz = np.load(os.path.join(data_dir, 'metathesaurus', 'test.npz'))
  subj = np.concatenate((data['subj'], npz['subj']))
  rel = np.concatenate((data['rel'], npz['rel']))
  obj = np.concatenate((data['obj'], npz['obj']))
  npz.close()
  print('read all data')
  valid_triples = set()
  for s, r, o in zip(subj, rel, obj):
    valid_triples.add((s, r, o))
  subj = np.asarray([s for s, _, _ in valid_triples])
  rel = np.asarray([r for _, r, _ in valid_triples])
  obj = np.asarray([o for _, _, o in valid_triples])

  perm = np.random.permutation(np.arange(len(rel)))
  num_test = 100000
  test_idx = perm[:num_test]
  train_idx = perm[num_test:]
  print('created train/test splits (%d/%d)' % (len(train_idx), len(test_idx)))

  np.savez_compressed(os.path.join(data_dir, 'metathesaurus', 'train.npz'),
                      subj=subj[train_idx],
                      rel=rel[train_idx],
                      obj=obj[train_idx])
  print('saved train set')

  np.savez_compressed(os.path.join(data_dir, 'metathesaurus', 'test.npz'),
                      subj=subj[test_idx],
                      rel=rel[test_idx],
                      obj=obj[test_idx])
  print('saved test set')


if __name__ == "__main__":
  if len(sys.argv) > 2:
    from_train_file(sys.argv[1], sys.argv[2], int(sys.argv[3]))
  else:
    from_train_file(sys.argv[1])
