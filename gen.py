'''Other comments: https://github.com/deepsound-project/samplernn-pytorch/issues/8'''
import torch

from collections import OrderedDict
import os
# import json
import numpy as np


from model import Runner, SampleRNN
from gen import Gen

from trainer.plugins import GeneratorPlugin

# Paths
RESULTS_PATH = '/tmp' #;results/exp:TEST-frame_sizes:16,4-n_rnn:2-dataset:COGNIMUSE_eq_eq_pad/'
PRETRAINED_PATH = os.path.abspath('results/NTS16k_3tier')
samples_path = os.path.join(RESULTS_PATH, 'generated')
if not os.path.exists(samples_path):
    os.mkdir(samples_path)


params = {
#   # model parameters
    'n_rnn': 3,
    'dim': 1024,
    'learn_h0': False,
    'q_levels': 256,
#   'seq_len': 1024,
    'weight_norm': True,
#   'batch_size': 128,
#   'val_frac': 0.1,
#   'test_frac': 0.1,
    'frame_sizes': [ 16, 16, 4 ],

#   # training parameters
#   'keep_old_checkpoints': False,
#   'datasets_path': 'datasets',
#   'results_path': 'results',
#   'learning_rate': 0.001,
#   'epoch_limit': 1000,
#   'resume': True,
    'sample_rate': 16000,
    'n_samples': 2,
    'sample_length':  16000,
#   'loss_smoothing': 0.99,
    'cuda': True,
#   'comet_key': None
}

tmp_pretrained_state = torch.load(PRETRAINED_PATH, map_location=lambda storage, loc: storage)
# Load all tensors onto GPU 1
# torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))

pretrained_state = OrderedDict()

for k, v in tmp_pretrained_state.items():
    # Delete "model." from key names since loading the checkpoint automatically attaches it
    layer_name = k.replace("model.", "")
    pretrained_state[layer_name] = v
    # print("k: {}, layer_name: {}, v: {}".format(k, layer_name, np.shape(v)))

# Create model with same parameters as used in training
model = SampleRNN(
    frame_sizes=params['frame_sizes'],
    n_rnn=params['n_rnn'],
    dim=params['dim'],
    learn_h0=params['learn_h0'],
    q_levels=params['q_levels'],
    weight_norm=params['weight_norm']
)
# Load pretrained model
model.load_state_dict(pretrained_state)

gen = Gen(Runner(model))
gen.register_plugin(GeneratorPlugin(
    samples_path, params['n_samples'],
    params['sample_length'], params['sample_rate']
))
gen.run();
exit(0)

# Generate Plugin
generator = GeneratorPlugin(GENERATED_PATH, params['n_samples'], params['sample_length'], params['sample_rate'])
kkkk

# Overloads register function to accept the trained model and the cuda setting
generator.register(model) #.cuda(), params['cuda'])

# Generate new audio
generator.epoch(1)
