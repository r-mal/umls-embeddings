from enum import Enum


class Policy(Enum):
  EPOCH = 1
  TIMED = 2


class ModelSaver:
  def __init__(self, tf_saver, session, model_file, policy):
    self.tf_saver = tf_saver
    self.session = session
    self.model_file = model_file
    self.policy = policy

  def save(self, global_step, policy, **kwargs):
    if self.save_condition(policy, **kwargs):
      print("Saving model to %s at step %d" % (self.model_file, global_step))
      self.tf_saver.save(self.session, self.model_file, global_step=global_step)

  def save_condition(self, policy, **kwargs):
    return policy == self.policy


class TimedSaver(ModelSaver):
  def __init__(self, tf_saver, session, model_file, seconds_per_save):
    ModelSaver.__init__(self, tf_saver, session, model_file, Policy.TIMED)
    self.seconds_per_save = seconds_per_save

  def save_condition(self, policy, **kwargs):
    return policy == self.policy and kwargs['seconds_since_last_save'] > self.seconds_per_save


class EpochSaver(ModelSaver):
  def __init__(self, tf_saver, session, model_file, save_every_x_epochs=1):
    ModelSaver.__init__(self, tf_saver, session, model_file, Policy.EPOCH)
    self.save_every_x_epochs = save_every_x_epochs

  def save_condition(self, policy, **kwargs):
    return 'epoch' in kwargs and kwargs['epoch'] % self.save_every_x_epochs == 0
