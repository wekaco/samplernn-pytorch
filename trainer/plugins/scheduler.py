from .plugin import Plugin
from torch.optim import lr_scheduler


class SchedulerPlugin(Plugin):

    def __init__(self, step_size):
        super().__init__([(1, 'epoch')])
        self.step_size = step_size
        self.verbose = True

        self.lr_scheduler = None

    def register(self, trainer):
        if self.step_size > 0:
            self.lr_scheduler = lr_scheduler.StepLR(
                trainer.optimizer._optimizer,
                step_size=self.step_size,
                verbose=self.verbose,
            )

    def epoch(self, epoch_index):
        if self.lr_scheduler:
            self.lr_scheduler.step()
