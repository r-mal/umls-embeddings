import os
import tensorflow as tf
import numpy as np
import math

from tf_util import Trainer, ModelSaver

from emb import EmbeddingModel
from gan import Generator, train_gan, Discriminator
import Config
from data import data_util, DataGenerator


def train():
  config = Config.flags

  if config.mode == 'gan':
    train_gan.train()
    exit()

  # init model dir
  config.model_dir = os.path.join(config.model_dir, config.model, config.run_name)
  if not os.path.exists(config.model_dir):
    os.makedirs(config.model_dir)

  # init summaries dir
  config.summaries_dir = os.path.join(config.summaries_dir, config.run_name)
  if not os.path.exists(config.summaries_dir):
    os.makedirs(config.summaries_dir)

  # save the config
  data_util.save_config(config.model_dir, config)

  # load data
  cui2id, data, train_idx, val_idx = data_util.load_metathesaurus_data(config.data_dir, config.val_proportion)
  config.val_progress_update_interval = int(math.floor(float(len(val_idx)) / config.batch_size))
  config.batches_per_epoch = int(math.floor(float(len(train_idx)) / config.batch_size))
  if not config.no_semantic_network:
    type2cuis = data_util.load_semantic_network_data(config.data_dir, data)
  else:
    type2cuis = None
  data_generator = DataGenerator.DataGenerator(data, train_idx, val_idx, config, type2cuis)

  # config map
  config_map = config.flag_values_dict()
  config_map['data'] = data
  config_map['train_idx'] = train_idx
  config_map['val_idx'] = val_idx
  if not config_map['no_semantic_network']:
    config_map['type2cuis'] = type2cuis

  with tf.Graph().as_default(), tf.Session() as session:
    # init model
    with tf.variable_scope(config.run_name):
      model = init_model(config, data_generator)
      # session.run(model.train_init_op)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # init saver
    tf_saver = tf.train.Saver(max_to_keep=10)
    saver = init_saver(config, tf_saver, session)

    # load model
    global_step = 0
    if config.load:
      ckpt = tf.train.latest_checkpoint(config.model_dir)
      print('Loading checkpoint: %s' % ckpt)
      global_step = int(os.path.split(ckpt)[-1].split('-')[-1])
      tf_saver.restore(session, ckpt)

    # finalize graph
    tf.get_default_graph().finalize()

    # define normalization step
    def find_unique(tensor_list):
      if max([len(t.shape) for t in tensor_list[:10]]) == 1:
        return np.unique(np.concatenate(tensor_list[:10]))
      else:
        return np.unique(np.concatenate([t.flatten() for t in tensor_list[:10]]))
    normalize = lambda _, batch: session.run(model.norm_op,
                                             {model.ids_to_update: find_unique(batch)})

    # define streaming_accuracy reset per epoch
    print('local variables that will be reinitialized every epoch: %s' % tf.local_variables())
    reset_local_vars = lambda: session.run(model.reset_streaming_metrics_op)

    # train
    Trainer.train(config_map, session, model, saver,
                  train_post_step=[normalize],
                  train_post_epoch=[reset_local_vars],
                  val_post_epoch=[reset_local_vars],
                  global_step=global_step,
                  max_batches_per_epoch=config_map['max_batches_per_epoch'])


def init_model(config, data_generator):
  print('Initializing %s embedding model in %s mode...' % (config.model, config.mode))
  npz = np.load(config.embedding_file) if config.load_embeddings else None

  if config.model == 'transe':
    em = EmbeddingModel.TransE(config, embeddings_dict=npz)
  elif config.model == 'transd':
    config.embedding_size = config.embedding_size / 2
    em = EmbeddingModel.TransD(config, embeddings_dict=npz)
  elif config.model == 'distmult':
    em = EmbeddingModel.DistMult(config, embeddings_dict=npz)
  else:
    raise ValueError('Unrecognized model type: %s' % config.model)

  if config.mode == 'disc':
    model = Discriminator.BaseModel(config, em, data_generator)
  elif config.mode == 'gen':
    model = Generator.Generator(config, em, data_generator)
  else:
    raise ValueError('Unrecognized mode: %s' % config.mode)

  if npz:
    # noinspection PyUnresolvedReferences
    npz.close()

  model.build()
  print('Built model.')
  print('use semnet: %s' % model.use_semantic_network)
  return model


def init_saver(config, tf_saver, session):
    model_file = os.path.join(config.model_dir, config.model)
    if config.save_strategy == 'timed':
      print('Models will be saved every %d seconds' % config.save_interval)
      return ModelSaver.TimedSaver(tf_saver, session, model_file, config.save_interval)
    elif config.save_strategy == 'epoch':
      print('Models will be saved every training epoch')
      return ModelSaver.EpochSaver(tf_saver, session, model_file)
    else:
      raise ValueError('Unrecognized save strategy: %s' % config.save_strategy)


if __name__ == "__main__":
  train()
