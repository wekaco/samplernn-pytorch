# wekaco/samplernn-pytorch

Fork of [deepsound-project/samplernn-pytorch](https://github.com/deepsound-project/samplernn-pytorch) "A PyTorch implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837)"
- Using pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2
- Docker ready

## Training

Prepare a dataset yourself. It should be a directory in `datasets/` filled with equal-length wav files. Or you can create your own dataset format by subclassing `torch.utils.data.Dataset`. It's easy, take a look at `dataset.FolderDataset` in this repo for an example.

The results - training log, loss plots, model checkpoints and generated samples will be saved in `results/`.

## Special thanks to

Continue the work of:
- [deepsound-project](https://github.com/deepsound-project)
- [Cortexelus](https://github.com/Cortexelus/)
- [khiner](https://github.com/khiner/)
- [gcunhase](https://github.com/gcunhase/)
- [depperpop](https://github.com/akx/depperpop)
- [akx](https://github.com/akx/)
