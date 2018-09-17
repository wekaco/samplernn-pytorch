from model import SampleRNN, Predictor
import torch
from torch.autograd import Variable
from collections import OrderedDict
import os
import json
from trainer.plugins import GeneratorPlugin
import numpy as np
from train import make_data_loader

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


# Get test data (source: train.py)
data_loader = make_data_loader(model.lookback, params)
test_split = 1 - params['test_frac']
val_split = test_split - params['val_frac']

dataset = data_loader(0, val_split, eval=False)
dataset_val = data_loader(0, test_split, eval=False)
dataset_test = data_loader(0, params['test_frac'], eval=False)


def wrap(input):
    if torch.is_tensor(input):
        input = Variable(input)
        if params['cuda']:
            input = input.cuda()
    return input


for data in dataset:
    batch_inputs = data[: -1]
    batch_target = data[-1]
    batch_inputs = list(map(wrap, batch_inputs))

    batch_target = Variable(batch_target)
    if params['cuda']:
        batch_target = batch_target.cuda()

    predictor = Predictor(model)
    if params['cuda']:
        model = model.cuda()
        predictor = predictor.cuda()

    prediction = predictor(*batch_inputs)  # , reset=False)
    prediction_data = prediction.data
    print(prediction)

    # Predict audios from 1 samples
    break
