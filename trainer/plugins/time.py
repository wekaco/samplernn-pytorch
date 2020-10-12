import time

from .monitor import Monitor


class AbsoluteTimeMonitor(Monitor):
    stat_name = 'time'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('unit', 's')
        kwargs.setdefault('precision', 0)
        kwargs.setdefault('running_average', False)
        kwargs.setdefault('epoch_average', False)
        super(AbsoluteTimeMonitor, self).__init__(*args, **kwargs)
        self.start_time = None

    def _get_value(self, *args):
        if self.start_time is None:
            self.start_time = time.time()
        return time.time() - self.start_time


