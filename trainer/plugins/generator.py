import os

from librosa.output import write_wav

from utils import sample_file_path
from model import Generator
from .plugin import Plugin


class GeneratorPlugin(Plugin):

    def __init__(self, samples_path, n_samples, sample_length, sample_rate, sampling_temperature=0.9, upload=None):
        super().__init__([(1, 'epoch')])
        self.samples_path = samples_path
        self.n_samples = n_samples
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self._upload = upload

        self.sampling_temperature = sampling_temperature

    def register(self, trainer):
        self.generate = Generator(trainer.model.model, trainer.cuda)
        self.trainer = trainer
        # print(self.trainer.stats)

    def epoch(self, epoch_index):
        samples = self.generate(self.n_samples, self.sample_length, self.sampling_temperature) \
                      .cpu().float().numpy()
        print("__epoch__");
        # print(self.trainer.stats)
        for i in range(self.n_samples):
            file_path = os.path.join(
                self.samples_path, 'audio{}-{}.wav'.format(epoch_index, i) #sample_file_path(epoch_index, self.trainer.iterations, self.trainer.stats["training_loss"]["last"].tolist(), i)
            )
            write_wav(
                file_path,
                samples[i, :],
                sr=self.sample_rate,
                norm=True
            )
            if self._upload is not None:
                self._upload(file_path)
