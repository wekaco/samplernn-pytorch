from model import SampleRNN
import torch
from collections import OrderedDict
import os
import json
from trainer.plugins import GeneratorPlugin


'''Other comments: https://github.com/deepsound-project/samplernn-pytorch/issues/8'''


# Paths
RESULTS_PATH = 'results/exp:TEST-frame_sizes:16,4-n_rnn:2-dataset:COGNIMUSE_eq_eq_pad/'
PRETRAINED_PATH = RESULTS_PATH + 'checkpoints/best-ep65-it79430'
GENERATED_PATH = RESULTS_PATH + 'generated/'
if not os.path.exists(GENERATED_PATH):
    os.mkdir(GENERATED_PATH)

# Load model parameters from .json for audio generation
params_path = RESULTS_PATH + 'sample_rnn_params.json'
with open(params_path, 'r') as fp:
    params = json.load(fp)

# Create model with same parameters as used in training
model = SampleRNN(
    frame_sizes=params['frame_sizes'],
    n_rnn=params['n_rnn'],
    dim=params['dim'],
    learn_h0=params['learn_h0'],
    q_levels=params['q_levels'],
    weight_norm=params['weight_norm']
)

# Delete "model." from key names since loading the checkpoint automatically attaches it to the key names
pretrained_state = torch.load(PRETRAINED_PATH)
new_pretrained_state = OrderedDict()

for k, v in pretrained_state.items():
    layer_name = k.replace("model.", "")
    new_pretrained_state[layer_name] = v
    # print("k: {}, layer_name: {}, v: {}".format(k, layer_name, np.shape(v)))

# Load pretrained model
model.load_state_dict(new_pretrained_state)

# Generate Plugin
generator = GeneratorPlugin(GENERATED_PATH, 2, params['sample_length'], params['sample_rate'])
# generator = GeneratorPlugin(generated_path, params['n_samples'], params['sample_length'], params['sample_rate'])

# Overloads register function to accept the trained model and the cuda setting
generator.register(model.cuda(), params['cuda'])

# Generate new audio
generator.epoch(1)
