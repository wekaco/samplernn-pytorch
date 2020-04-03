# wekaco/samplernn-pytorch

.
Fork of [deepsound-project/samplernn-pytorch](https://github.com/deepsound-project/samplernn-pytorch) "A PyTorch implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837)"
- Using torch 0.4.1 (see issues for warnings)
- Docker ready

## Training

Prepare a dataset yourself. It should be a directory in `datasets/` filled with equal-length wav files. Or you can create your own dataset format by subclassing `torch.utils.data.Dataset`. It's easy, take a look at `dataset.FolderDataset` in this repo for an example.

The results - training log, loss plots, model checkpoints and generated samples will be saved in `results/`.
