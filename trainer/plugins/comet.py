import os

from .plugin import Plugin
from utils import sample_file_path


class CometPlugin(Plugin):
    # pattern = 'ep{}-s{}.wav'

    def __init__(self, experiment, fields, samples_path, n_samples, sample_rate):
        super().__init__([(1, 'epoch')])

        self.experiment = experiment
        self.fields = [
            field if type(field) is tuple else (field, 'last')
            for field in fields
        ]
        self.n_samples = n_samples
        self.samples_path = samples_path
        self.sample_rate = sample_rate

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, epoch_index):
        for (field, stat) in self.fields:
            self.experiment.log_metric(field, self.trainer.stats[field][stat])
        self.experiment.log_epoch_end(epoch_index)
        for i in range(self.n_samples):
            self.experiment.log_audio(
                os.path.join(
                    self.samples_path, 'audio{}-{}.wav'.format(epoch_index, i) # sample_file_path(epoch_index, self.trainer.iterations, self.trainer.stats["training_loss"]["last"].tolist(), i)
                ),
                sample_rate=self.sample_rate
            )
