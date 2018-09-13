class Trainable:
  def __init__(self):
    pass

  def fetches(self, is_training, verbose=False):
    """
    Returns a list of fetches to be passed to session.run()
    :param is_training: flag indicating if the model is training/testing
    :param verbose: flag indicating if a more verbose set of variables should be fetched (usually for debugging or
                    progress updates)
    :return: a list of fetches to be passed to session.run()
    """
    raise NotImplementedError("to be implemented by subclass")

  def prepare_feed_dict(self, batch, is_training, **kwargs):
    """
    Turns a list of tensors into a dict of model parameter: tensor
    :param batch: list of data tensors to be passed to the model
    :param is_training: flag indicating if the model is in training or testing mode
    :param kwargs: optional other params
    :return: the feed dict to be passed to session.run()
    """
    raise NotImplementedError("to be implemented by subclass")

  def progress_update(self, batch, fetched, **kwargs):
    """
    Prepares a progress update
    :param batch: batch data passed to prepare_feed_dict()
    :param fetched: tensors returned by session.run()
    :param kwargs: optional other params
    :return: String progress update
    """
    raise NotImplementedError("to be implemented by subclass")

  def data_provider(self, config, is_training, **kwargs):
    """
    Provides access to a data generator that generates batches of data
    :param config: dict of config flags to values (usually tf.flags.FLAGS)
    :param is_training: flag indicating if the model is in training or testing mode
    :param kwargs: optional other params
    :return: A generator that generates batches of data
    """
    raise NotImplementedError("to be implemented by subclass")
