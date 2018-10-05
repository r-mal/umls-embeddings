import tensorflow as tf

from tensorflow.contrib import layers


class BaseModel:
  def __init__(self, config):
    self.embedding_size = config.embedding_size
    self.embedding_device = config.embedding_device
    self.vocab_size = config.vocab_size

    self.embeddings = None
    self.ids_to_update = tf.placeholder(dtype=tf.int32, shape=[None], name='used_vectors')
    self.norm_op = None

  def energy(self, head, rel, tail, norm_ord='euclidean'):
    """
    Calculates the energies of the batch of triples corresponding to head, rel and tail.
    Energy should be in (-inf, 0]
    :param head: [batch_size] vector of head entity ids
    :param rel: [batch_size] vector of relation ids
    :param tail: [batch_size] vector of tail entity ids
    :param norm_ord: order of the normalization function
    :return: [batch_size] vector of energies of the passed triples
    """
    raise NotImplementedError("subclass should implement")

  def embedding_lookup(self, ids):
    """
    returns embedding vectors or tuple of embedding vectors for the passed ids
    :param ids: ids of embedding vectors in an embedding matrix
    :return: embedding vectors or tuple of embedding vectors for the passed ids
    """
    raise NotImplementedError("subclass should implement")

  def normalize_parameters(self):
    """
    Returns the op that enforces the constraint that each embedding vector have a norm <= 1
    :return: the op that enforces the constraint that each embedding vector have a norm <= 1
    """
    raise NotImplementedError("subclass should implement")

  # noinspection PyMethodMayBeStatic
  def normalize(self, rows, mat, ids):
    """
    Normalizes the rows of the matrix mat corresponding to ids s.t. |row|_2 <= 1 for each row.
    :param rows: rows from the matrix mat, embedding vectors
    :param mat: matrix of embedding vectors
    :param ids: ids of rows in mat
    :return: the scatter update op that updates only the rows of mat corresponding to ids
    """
    norm = tf.norm(rows)
    scaling = 1. / tf.maximum(norm, 1.)
    scaled = scaling * rows
    return tf.scatter_update(mat, ids, scaled)


class TransE(BaseModel):
  def __init__(self, config, embeddings_dict=None):
    BaseModel.__init__(self, config)
    with tf.device("/%s:0" % self.embedding_device):
      if embeddings_dict is None:
        self.embeddings = tf.get_variable("embeddings",
                                          shape=[self.vocab_size, self.embedding_size],
                                          dtype=tf.float32,
                                          initializer=layers.xavier_initializer())
      else:
        self.embeddings = tf.Variable(embeddings_dict['embs'], name="embeddings")

  def energy(self, head, rel, tail, norm_ord='euclidean'):
    h = self.embedding_lookup(head)
    r = self.embedding_lookup(rel)
    t = self.embedding_lookup(tail)

    return tf.norm(h + r - t,
                   ord=norm_ord,
                   axis=-1,
                   keepdims=False,
                   name='energy')

  def embedding_lookup(self, ids):
    with tf.device("/%s:0" % self.embedding_device):
      return tf.nn.embedding_lookup(self.embeddings, ids)

  def normalize_parameters(self):
    """
    Enforces the contraint that the embedding vectors corresponding to ids_to_update <= 1.0
    """
    params1 = self.embedding_lookup(self.ids_to_update)
    self.norm_op = self.normalize(params1, self.embeddings, self.ids_to_update)

    return self.norm_op


class TransD(BaseModel):
  def __init__(self, config, embeddings_dict=None):
    BaseModel.__init__(self, config)
    with tf.device("/%s:0" % self.embedding_device):
      if embeddings_dict is None:
        print('Initializing embeddings.')
        self.embeddings = tf.get_variable("embeddings",
                                          shape=[self.vocab_size, self.embedding_size],  # 364373
                                          dtype=tf.float32,
                                          initializer=layers.xavier_initializer())
      else:
        print('Loading embeddings.')
        self.embeddings = tf.Variable(embeddings_dict['embs'], name="embeddings")
    with tf.device("/%s:%d" % (self.embedding_device, 0 if self.embedding_device == 'cpu' else 0)):
      if embeddings_dict is None or 'p_embs' not in embeddings_dict:
        print('Initializing projection embeddings.')
        if config.p_init == 'zeros':
          p_init = tf.initializers.zeros()
        elif config.p_init == 'xavier':
          p_init = layers.xavier_initializer()
        elif config.p_init == 'uniform':
          p_init = tf.initializers.random_uniform(minval=-0.1, maxval=0.1, dtype=tf.float32)
        else:
          raise Exception('unrecognized p initializer: %s' % config.p_init)

        # projection embeddings initialized to zeros
        self.p_embeddings = tf.get_variable("p_embeddings",
                                            shape=[self.vocab_size, self.embedding_size],
                                            dtype=tf.float32,
                                            initializer=p_init)
      else:
        print('Loading projection embeddings.')
        self.p_embeddings = tf.Variable(embeddings_dict['p_embs'], name="p_embeddings")

  def energy(self, head, rel, tail, norm_ord='euclidean'):
    """
        Computes the TransD energy of a relation triple
        :param head: head concept embedding ids [batch_size]
        :param rel: relation embedding ids [batch_size]
        :param tail: tail concept embedding ids [batch_size]
        :param norm_ord: norm order ['euclidean', 'fro', 'inf', 1, 2, 3, etc.]
        :return: [batch_size] vector of energies
        """
    # x & x_proj both [batch_size, embedding_size]
    h, h_proj = self.embedding_lookup(head)
    r, r_proj = self.embedding_lookup(rel)
    t, t_proj = self.embedding_lookup(tail)

    # [batch_size]
    return tf.norm(self.project(h, h_proj, r_proj) + r - self.project(t, t_proj, r_proj),
                   ord=norm_ord,
                   axis=1,
                   keepdims=False,
                   name="energy")

  # noinspection PyMethodMayBeStatic
  def project(self, c, c_proj, r_proj):
    """
    Computes the projected concept embedding for relation r according to TransD:
      (c_proj^T*c)*r_proj + c
    :param c: concept embeddings [batch_size, embedding_size]
    :param c_proj: concept projection embeddings [batch_size, embedding_size]
    :param r_proj: relation projection embeddings [batch_size, embedding_size]
    :return: projected concept embedding [batch_size, embedding_size]
    """
    return c + tf.reduce_sum(c * c_proj, axis=-1, keepdims=True) * r_proj

  def embedding_lookup(self, ids):
    with tf.device("/%s:0" % self.embedding_device):
      params1 = tf.nn.embedding_lookup(self.embeddings, ids)
    with tf.device("/%s:%d" % (self.embedding_device, 1 if self.embedding_device == 'cpu' else 0)):
      params2 = tf.nn.embedding_lookup(self.p_embeddings, ids)
    return params1, params2

  def normalize_parameters(self):
    """
    Normalizes the vectors of embeddings corresponding to the passed ids
    :return: the normalization op
    """
    with tf.device("/%s:0" % self.embedding_device):
      params1 = tf.nn.embedding_lookup(self.embeddings, self.ids_to_update)
    with tf.device("/%s:%d" % (self.embedding_device, 1 if self.embedding_device == 'cpu' else 0)):
      params2 = tf.nn.embedding_lookup(self.p_embeddings, self.ids_to_update)

    n1 = self.normalize(params1, self.embeddings, self.ids_to_update)
    n2 = self.normalize(params2, self.p_embeddings, self.ids_to_update)
    self.norm_op = n1, n2

    return self.norm_op


class DistMult(TransE):
  def __init__(self, config, embeddings_dict=None):
    BaseModel.__init__(self, config)
    if config.energy_activation == 'relu':
      self.energy_activation = tf.nn.relu
    elif config.energy_activation == 'tanh':
      self.energy_activation = tf.nn.tanh
    elif config.energy_activation == 'sigmoid':
      self.energy_activation = tf.nn.sigmoid
    elif config.energy_activation is None:
      self.energy_activation = lambda x: x
    else:
      raise Exception('Unrecognized activation: %s' % config.energy_activation)

    with tf.device("/%s:0" % self.embedding_device):
      if embeddings_dict is None:
        self.embeddings = tf.get_variable("embeddings",
                                          shape=[self.vocab_size, self.embedding_size],
                                          dtype=tf.float32,
                                          initializer=tf.initializers.random_uniform(minval=-0.5,
                                                                                     maxval=0.5,
                                                                                     dtype=tf.float32))
      else:
        self.embeddings = tf.Variable(embeddings_dict['embs'], name="embeddings")

  def energy(self, head, rel, tail, norm_ord='euclidean'):
    h = self.embedding_lookup(head)
    r = self.embedding_lookup(rel)
    t = self.embedding_lookup(tail)

    return self.energy_activation(tf.reduce_sum(h * r * t,
                                                axis=-1,
                                                keepdims=False))

  def normalize_parameters(self):
    return tf.no_op()

  def regularization(self, parameters):
    reg_term = 0
    for p in parameters:
      reg_term += tf.reduce_sum(tf.norm(self.embedding_lookup(p)))
    return reg_term
