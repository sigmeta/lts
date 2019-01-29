

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf
import numpy as np

from tensor2tensor.utils import trainer_utils as utils
from tensor2tensor.visualization import attention
from tensor2tensor.utils import decoding
from tensor2tensor.utils import usr_dir

from tensor2tensor.layers import common_layers
import os
# PUT THE MODEL YOU WANT TO LOAD HERE!



flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("t2t_usr_dir", "",
                      "Path to a Python module that will be imported. The ")
flags.DEFINE_string("output_dir", "", "Base output directory for run.")

usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
FLAGS.schedule = 'train_and_evaluate'




hparams = utils.create_hparams(FLAGS.hparams_set, FLAGS.data_dir)

# SET EXTRA HYPER PARAMS HERE!
#hparams.null_slot = True

utils.add_problem_hparams(hparams, FLAGS.problems)

num_datashards = utils.devices.data_parallelism().n

mode = tf.estimator.ModeKeys.EVAL

input_fn = utils.input_fn_builder.build_input_fn(
      mode=mode,
      hparams=hparams,
      data_dir=FLAGS.data_dir,
      num_datashards=num_datashards,
      worker_replicas=FLAGS.worker_replicas,
      worker_id=FLAGS.worker_id,
      batch_size=32)

inputs, target = input_fn()
features = inputs
features['targets'] = target




def decode(ids):
    return hparams.problems[0].vocabulary['targets'].decode(np.squeeze(ids))

def to_tokens(ids):
    ids = np.squeeze(ids)
    subtokenizer = hparams.problems[0].vocabulary['targets']
    tokens = []
    for _id in ids:
        if _id == 0:
            tokens.append('<PAD>')
        elif _id == 1:
            tokens.append('<EOS>')
        elif _id == 2:
            tokens.append('<SRC_EOS>')
        else:
            tokens.append(subtokenizer._subtoken_id_to_subtoken_string(_id))
    return tokens

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True) # only difference



model_fn=utils.model_builder.build_model_fn(
    FLAGS.model,
    problem_names=[FLAGS.problems],
    train_steps=FLAGS.train_steps,
    worker_id=FLAGS.worker_id,
    worker_replicas=FLAGS.worker_replicas,
    eval_run_autoregressive=FLAGS.eval_run_autoregressive,
    decode_hparams=decoding.decode_hparams(FLAGS.decode_hparams))
est_spec = model_fn(features, target, mode, hparams)



loss, weight = common_layers.padded_cross_entropy(
         est_spec.predictions['predictions'],
         target,
         0,
	 reduce_sum=False)




with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    beam_out = model_fn(features, target, tf.contrib.learn.ModeKeys.INFER, hparams)


sv = tf.train.Supervisor(
    logdir=FLAGS.output_dir,
    global_step=tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step'))
sess = sv.PrepareSession(config=tf.ConfigProto(allow_soft_placement=True))
sv.StartQueueRunners(
    sess,
    tf.get_default_graph().get_collection(tf.GraphKeys.QUEUE_RUNNERS))


loss_sum = 0.0
token_sum = 0.0
sentence_cnt = 0 
try:
    while not sv.should_stop(): 
        minibatch_loss, minibatch_weight = sess.run([loss, weight])
        
        sentence_cnt += minibatch_loss.shape[0] 
        loss_sum += np.sum(minibatch_loss)
        token_sum += np.sum(minibatch_weight)
except Exception as e:
    print(e.message)

print('sentence_cnt: ', sentence_cnt)
print('token_sum: ', token_sum)
print('valid_loss: ', loss_sum*1.0/token_sum)

