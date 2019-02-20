import tensorflow as tf
import six
import functools

def _example_length(example):
  length = 0
  # Length of the example is the maximum length of the feature lengths
  for v in example.values():
    # For images the sequence length is the size of the spatial dimensions.
    feature_length = (tf.shape(v)[0] if len(v.get_shape()) < 3 else
                      tf.shape(v)[0] * tf.shape(v)[1])
    length = tf.maximum(length, feature_length)
  return length
def example_valid_size(example, min_length, max_length):
  length = _example_length(example)
  return tf.logical_and(
      length >= min_length,
      length <= max_length,
  )

dataset=tf.contrib.data.TFRecordDataset("lts_data/lts-train-00000-of-00001")
#tensor1=dataset.make_one_shot_iterator().get_next()
data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64),
        "teachers": tf.VarLenFeature(tf.float32)
    }
data_items_to_decoders = {
          field: tf.contrib.slim.tfexample_decoder.Tensor(field)
          for field in data_fields
      }

print("data_items_to_decoders",data_items_to_decoders["teachers"])
def cast_int64_to_int32(features):
  f = {}
  for k, v in six.iteritems(features):
    if v.dtype == tf.int64:
      v = tf.to_int32(v)
    f[k] = v
  return f
def decode_record(record):
    """Serialized Example to dict of <feature name, Tensor>."""
    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        data_fields, data_items_to_decoders)

    print("decoder",decoder)
    decode_items = list(data_items_to_decoders)
    decoded = decoder.decode(record, items=decode_items)
    print(decoded[2])
    #decoded[0] = tf.Print(decoded[0], [decoded[0]], "decoded", summarize=1000)
    #decoded[1] = tf.Print(decoded[1], [decoded[1]], "decoded", summarize=1000)
    #decoded[2] = tf.Print(decoded[2], [decoded[2]], "decoded", summarize=1000)
    return dict(zip(decode_items, decoded))

dataset = dataset.map(decode_record)
dataset = dataset.map(cast_int64_to_int32)
dataset = dataset.filter(
        functools.partial(
            example_valid_size,
            min_length=0,
            max_length=256,
        ))

dataset = dataset.shuffle(10000)
dataset = dataset.repeat(None)
itr=dataset.make_one_shot_iterator()
tensor1 = itr.get_next()
tensor = itr.get_next()
tensor["teachers"]=tf.reshape(tensor["teachers"],[-1,76])
with tf.Session() as sess:
    a = sess.run(tensor1)
    print(a)
    a=sess.run(tensor)
    print(a)




