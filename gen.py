'''Other comments: https://github.com/deepsound-project/samplernn-pytorch/issues/8'''
import argparse

import torch

from collections import OrderedDict
import os
# import json
import numpy as np


from model import Runner, SampleRNN
from gen import Gen

from trainer.plugins import GeneratorPlugin

from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket

# Imports the Google Cloud client library
import logging
from google.cloud.logging.resource import Resource
from google.cloud.logging.handlers import CloudLoggingHandler
import google.cloud.logging # Don't conflict with standard logging

from uuid import uuid4

def setup_logging(name):
    res = Resource(
        type="generic_task",
        labels={
            "location": "vast.ai",
            "task_id": str(uuid4()),
            "namespace": "samplernn-pytorch",
            "job": "gen.py",
        },
    )
    # Instantiates a client
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client, name, resource=res)
    
    google.cloud.logging.handlers.setup_logging(handler)

GCS_BUCKET='samplern'

def main(checkpoint, **args):
    params = {
        'n_rnn': 3,
        'dim': 1024,
        'learn_h0': False,
        'q_levels': 256,
        'weight_norm': True,
        'frame_sizes': [ 16, 16, 4 ],
        'sample_rate': 16000,
        'n_samples': 1,
        'sample_length':  16000,
        'cuda': True,
    }
    logging.info('booting')


    # dataset = storage_client.list_blobs(bucket, prefix=path)
    # for blob in dataset:
    #   blob.download_to_filename(blob.name)
    bucket = None

    checkpoint = os.path.abspath(checkpoint)

    if params['bucket']:
        storage_client = storage.Client()
        bucket = Bucket(storage_client, GCS_BUCKET)
    
    # Paths
    RESULTS_PATH = '/tmp' #;results/exp:TEST-frame_sizes:16,4-n_rnn:2-dataset:COGNIMUSE_eq_eq_pad/'
    PRETRAINED_PATH = os.path.abspath('results/NTS16k_3tier')
    samples_path = os.path.join(RESULTS_PATH, 'generated')
    if not os.path.exists(samples_path):
        os.mkdir(samples_path)

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

    def upload(file_path):
        if bucket is None:
            return

        name = file_path.replace(os.path.abspath(os.curdir) + '/', '')
        blob = Blob(name, bucket)
        blob.upload_from_filename(file_path)

    gen = Gen(Runner(model))
    gen.register_plugin(GeneratorPlugin(
        samples_path, params['n_samples'],
        params['sample_length'], params['sample_rate'],
        upload
    ))
    gen.run();

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )

    def parse_bool(arg):
        arg = arg.lower()
        if 'true'.startswith(arg):
            return True
        elif 'false'.startswith(arg):
            return False
        else:
            raise ValueError()

    setup_logging('gen')

    parser.add_argument(
        '--bucket',
        help='google cloud storage bucket name for datasets and results',
        default=None
    )
    parser.add_argument(
        '--checkpoint',
        help='experiment name',
        required=True
    )
    parser.add_argument(
        '--cuda', type=parse_bool,
        help='whether to use CUDA'
    )
    #parser.set_defaults(**default_params)

    main(**vars(parser.parse_args()))
