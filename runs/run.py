#!/home/xutan/anaconda3/bin/python

from philly.philly_tool import Philly

# c = 'cam'
c = 'rr1'
philly = Philly({
    'cluster_id': c,
    'debug': True,
    'gpus': 1,
})
philly.prepare_submit()


'''
for num_layers in [2, 4, 6]:
  for hidden_size in [128, 256, 512]:
    for num_heads in [4, 8, 16]:
      for batch_size in [2048, 4096, 8192]:
        exp_name = 'lts_l{}_h{}_he{}_b{}'.format(num_layers, hidden_size, num_heads, batch_size)
        philly.add_task(
        'HPARAMS="num_decoder_layers={},num_encoder_layers={},hidden_size={},filter_size={},num_heads={},batch_size={}," SETTING={} bash runs/train_lts.sh'
          .format(num_layers, num_layers, hidden_size, hidden_size*4, num_heads, batch_size, exp_name),
        exp_name=exp_name,
        gpus=1,
        )
'''

for num_layers in [2, 4, 6]:
  for hidden_size in [128, 256, 512]:
    for num_heads in [4, 8, 16]:
      for batch_size in [2048, 4096, 8192]:
        exp_name = 'lts_l{}_h{}_he{}_b{}'.format(num_layers, hidden_size, num_heads, batch_size)
        philly.add_task(
        'HPARAMS="num_decoder_layers={},num_encoder_layers={},hidden_size={},filter_size={},num_heads={},batch_size={}," SETTING={} bash runs/test_lts.sh'
          .format(num_layers, num_layers, hidden_size, hidden_size*4, num_heads, batch_size, exp_name),
        exp_name="test@"+exp_name,
        gpus=1,
        )



philly.start()
