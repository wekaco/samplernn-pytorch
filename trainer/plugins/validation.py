import torch
from torch.autograd import Variable

from .plugin import Plugin


class ValidationPlugin(Plugin):

    def __init__(self, val_dataset, test_dataset):
        super().__init__([(1, 'epoch')])
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def register(self, trainer):
        self.trainer = trainer
        val_stats = self.trainer.stats.setdefault('validation_loss', {})
        val_stats['log_epoch_fields'] = ['{last:.4f}']
        test_stats = self.trainer.stats.setdefault('test_loss', {})
        test_stats['log_epoch_fields'] = ['{last:.4f}']

    def epoch(self, idx):
        self.trainer.model.eval()

        val_stats = self.trainer.stats.setdefault('validation_loss', {})
        val_stats['last'] = self._evaluate(self.val_dataset)
        test_stats = self.trainer.stats.setdefault('test_loss', {})
        test_stats['last'] = self._evaluate(self.test_dataset)

        self.trainer.model.train()

    def _evaluate(self, dataset):
        loss_sum = 0
        n_examples = 0
        for data in dataset:
            batch_inputs = data[: -1]
            batch_target = data[-1]
            batch_size = batch_target.size()[0]

            def wrap(input):
                if torch.is_tensor(input):
                    with torch.no_grad():
                        input = Variable(input)
                        if self.trainer.cuda:
                            input = input.cuda()
                return input

            batch_inputs = list(map(wrap, batch_inputs))

            with torch.no_grad():
                batch_target = Variable(batch_target)
                if self.trainer.cuda:
                    batch_target = batch_target.cuda()

                batch_output = self.trainer.model(*batch_inputs)
                loss_sum += self.trainer.criterion(batch_output, batch_target) \
                                        .data.item() * batch_size

            n_examples += batch_size

            return loss_sum / n_examples

        # patch division by zero for micro datasets
        return loss_sum
