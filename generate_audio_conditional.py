from model import SampleRNN, Predictor
import torch
from torch.autograd import Variable
from collections import OrderedDict
import os
import json
from trainer.plugins import GeneratorPlugin
import numpy as np
from train import make_data_loader
from dataset import FolderDataset, DataLoader
import utils


'''Other comments: https://github.com/deepsound-project/samplernn-pytorch/issues/8'''


# Paths
RESULTS_PATH = 'results/exp:TEST-frame_sizes:16,4-n_rnn:2-dataset:COGNIMUSE_eq_eq_pad/'
PRETRAINED_PATH = RESULTS_PATH + 'checkpoints/best-ep65-it79430'
# PRETRAINED_PATH = 'results/exp:TEST-frame_sizes:16,4-n_rnn:2-dataset:piano3/' + 'checkpoints/best-ep21-it29610'
GENERATED_PATH = RESULTS_PATH + 'generated_conditioned/'
if not os.path.exists(GENERATED_PATH):
    os.mkdir(GENERATED_PATH)


def make_data_loader2(overlap_len, params, dataset_path):
    path = os.path.join(params['datasets_path'], dataset_path)

    def data_loader(split_from, split_to, eval):
        dataset = FolderDataset(
            path, overlap_len, params['q_levels'], split_from, split_to
        )
        l = dataset.__len__()
        dataset_filenames = []
        for i in range(0, l):
            # print(dataset.get_filename(i))
            dataset_filenames.append(dataset.get_filename(i))
        dataloader = DataLoader(
            dataset,
            batch_size=params['batch_size'],
            seq_len=params['seq_len'],
            overlap_len=overlap_len,
            shuffle=(not eval),
            drop_last=(not eval)
        )
        return dataloader, dataset_filenames
    return data_loader


def get_test_data(model, params):
    # Get test data (source: train.py)
    data_loader = make_data_loader2(model.lookback, params, params['dataset'])
    test_split = 1 - params['test_frac']
    val_split = test_split - params['val_frac']

    dataset, dataset_filenames = data_loader(0, val_split, eval=False)
    dataset_val, dataset_val_filenames = data_loader(val_split, test_split, eval=False)
    dataset_test, dataset_test_filenames = data_loader(test_split, 1, eval=False)

    print("train: {}".format(dataset_filenames))
    print("test: {}".format(dataset_test_filenames))
    print("val: {}".format(dataset_val_filenames))

    # Get test data (source: train.py)
    # data_loader = make_data_loader2(model.lookback, params, params['dataset'])
    data_loader = make_data_loader2(model.lookback, params, 'Seminar_test')  # 'COGNIMUSE_eq_eq_pad_test')
    dataset, dataset_filenames = data_loader(0, 1, eval=False)

    print("single: {}".format(dataset_filenames))
    return dataset_filenames


if __name__ == '__main__':
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

    dataset_filenames = get_test_data(model, params)

    # Gets initial samples form 1 test sample and check if it re-generates it
    audio_filename = dataset_filenames[0]
    from librosa.core import load
    sr = params['sample_rate']
    seq, sr = load(audio_filename, sr=sr, mono=True)
    print("Sample rate: {}".format(sr))

    # Generate Plugin
    num_samples = 6  # params['n_samples']

    initial_seq_size = 64 * 100  # has to be multiple of rnn.n_frame_samples ???
    initial_seq = None
    if initial_seq_size > 1:
        init = utils.linear_quantize(torch.from_numpy(seq[0:initial_seq_size]), params['q_levels'])
        # init = seq[0:initial_seed_size]
        init = np.tile(init, (num_samples, 1))
        initial_seq = torch.LongTensor(init)
        # initial_seed = utils.linear_quantize(initial_seed, params['q_levels'])

    sample_length = params['sample_length']
    sample_rate = params['sample_rate']
    print("Number samples: {}, sample_length: {}, sample_rate: {}".format(num_samples, sample_length, sample_rate))
    generator = GeneratorPlugin(GENERATED_PATH, num_samples, sample_length, sample_rate)

    # Overloads register function to accept the trained model and the cuda setting
    generator.register_generate(model.cuda(), params['cuda'])

    # Generate new audio
    generator.epoch('Test19_{}'.format(initial_seq_size), initial_seed=initial_seq)
