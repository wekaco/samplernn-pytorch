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

# TODO: duplicate
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def setup_logging(name, log_level=20):
    task_id = str(uuid4())
    res = Resource(
        type="generic_task",
        labels={
            "location": "vast.ai",
            "task_id": task_id,
            "namespace": "samplernn-pytorch",
            "job": "gen.py",
        },
    )

    # Instantiates a client
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client, name, resource=res)
    
    google.cloud.logging.handlers.setup_logging(handler, log_level=log_level)
    return task_id

def preload_checkpoint(path, storage_client, bucket):
    if os.path.isfile(path):
        logging.debug('found local copy of {}'.format(path))
        return

    dir_name = os.path.abspath(os.path.join(path, os.pardir))
    ensure_dir_exists(dir_name)

    logging.info('searching {}'.format(path))
    dataset = storage_client.list_blobs(bucket, prefix=path)
    for blob in dataset:
        logging.info('downloading {}'.format(blob.name))

        blob.download_to_filename(blob.name)

def main(checkpoint, **args):
    task_id = setup_logging('gen', logging.NOTSET if args.get('debug', False) else logging.INFO)

    params = {
        'n_rnn': 3,
        'dim': 1024,
        'learn_h0': False,
        'q_levels': 256,
        'weight_norm': True,
        'frame_sizes': [ 16, 16, 4 ],
        'sample_rate': 16000,
        'n_samples': 10,
        'sample_length':  16000
    }
    logging.info('booting')

    # dataset = storage_client.list_blobs(bucket, prefix=path)
    # for blob in dataset:
    #   blob.download_to_filename(blob.name)
    bucket = None

    if args['bucket']:
        logging.debug('setup google storage bucket {}'.format(args['bucket']))
        storage_client = storage.Client()
        bucket = Bucket(storage_client, args['bucket'])

        preload_checkpoint(checkpoint, storage_client, bucket)

    results_path = os.path.abspath(os.path.join(checkpoint, os.pardir, os.pardir, task_id))
    ensure_dir_exists(results_path)

    checkpoint = os.path.abspath(checkpoint)

    tmp_pretrained_state = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(0) if args['cuda'] else storage)
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

        # remove prefix /app
        name = file_path.replace(os.path.abspath(os.curdir) + '/', '')
        blob = Blob(name, bucket)
        logging.info('uploading {}'.format(name))
        blob.upload_from_filename(file_path)

    gen = Gen(Runner(model.cuda() if args['cuda'] else model))
    gen.register_plugin(GeneratorPlugin(
        results_path, params['n_samples'],
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
        default='true',
        help='whether to use CUDA'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='debug mode'
    )
    #parser.set_defaults(**default_params)

    main(**vars(parser.parse_args()))