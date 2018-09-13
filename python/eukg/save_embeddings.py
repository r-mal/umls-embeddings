import tensorflow as tf
import os
import numpy as np

import Config
from train import init_model


def main():
  config = Config.flags

  if config.mode == 'gan':
    scope = config.dis_run_name
    run_name = config.run_name + "/discriminator"
    config.mode = 'disc'
  else:
    scope = config.run_name
    run_name = config.run_name

  with tf.Graph().as_default(), tf.Session() as session:
    with tf.variable_scope(scope):
      model = init_model(config, None)
      saver = tf.train.Saver([var for var in tf.global_variables() if 'embeddings' in var.name])
      ckpt = tf.train.latest_checkpoint(os.path.join(config.model_dir, config.model, run_name))
      print('Loading checkpoint: %s' % ckpt)
      saver.restore(session, ckpt)

      embeddings = session.run(model.embedding_model.embeddings)
      if config.model == 'transd':
        embeddings = np.concatenate((embeddings, session.run(model.embedding_model.p_embeddings)), axis=1)

      np.savez_compressed(config.embedding_file,
                          embs=embeddings)


if __name__ == "__main__":
  main()
