# samplernn-pytorch

A PyTorch implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837).

![A visual representation of the SampleRNN architecture](http://deepsound.io/images/samplernn.png)

It's based on the reference implementation in Theano: https://github.com/soroushmehr/sampleRNN_ICLR2017. Unlike the Theano version, our code allows training models with arbitrary number of tiers, whereas the original implementation allows maximum 3 tiers. However it doesn't allow using LSTM units (only GRU). For more details and motivation behind rewriting this model to PyTorch, see our blog post: http://deepsound.io/samplernn_pytorch.html.

## Dependencies

This code requires Python 3.5+ and PyTorch 0.1.12+ (try last three options below). Installation instructions for PyTorch are available on their website: http://pytorch.org/. You can install the rest of the dependencies by running `pip install -r requirements.txt`.
```bash
pip install -U numpy scipy matplotlib natsort
git clone https://github.com/librosa/librosa
cd librosa
python setup.py build
python setup.py install
pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36mu-linux_x86_64.whl
```
## Datasets

We provide a script for creating datasets from YouTube single-video mixes. It downloads a mix, converts it to wav and splits it into equal-length chunks. To run it you need youtube-dl (a recent version; the latest version from pip should be okay) and ffmpeg. To create an example dataset - 4 hours of piano music split into 8 second chunks, run:

```
cd datasets
./download-from-youtube.sh "https://www.youtube.com/watch?v=EhO_MrRfftU" 8 piano
```

You can also prepare a dataset yourself. It should be a directory in `datasets/` filled with equal-length wav files. Or you can create your own dataset format by subclassing `torch.utils.data.Dataset`. It's easy, take a look at `dataset.FolderDataset` in this repo for an example.

## Pre-processing
```
python audio_preprocessing.py --folder FOLDER_NAME
```

## Training

To train the model you need to run `train.py`. All model hyperparameters are settable in the command line. Most hyperparameters have sensible default values, so you don't need to provide all of them. Run `python train.py -h` for details. To train on the `piano` dataset using the best hyperparameters we've found, run:

```
CUDA_VISIBLE_DEVICES=0 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 3 --dataset piano3
CUDA_VISIBLE_DEVICES=1 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset COGNIMUSE_eq_eq_pad
CUDA_VISIBLE_DEVICES=2 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 3 --q_levels 512 --dataset COGNIMUSE_eq_eq_pad

CUDA_VISIBLE_DEVICES=0,1 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset splices_audio_BMI_16000_c1_16bits_music_eq
```

The results - training log, loss plots, model checkpoints and generated samples will be saved in `results/`.

We also have an option to monitor the metrics using [CometML](https://www.comet.ml/). To use it, just pass your API key as `--comet_key` parameter to `train.py`.
