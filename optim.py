from torch.nn.functional import hardtanh


def gradient_clipping(optimizer, min=-1, max=1):

    class OptimizerWrapper(object):
        def __init__(self, optimizer):
            self._optimizer = optimizer

        def step(self, closure):
            def closure_wrapper():
                loss = closure()
                for group in self._optimizer.param_groups:
                    for p in group['params']:
                        hardtanh(p.grad, min, max, inplace=True)
                return loss
            
            return self._optimizer.step(closure_wrapper)

        def __getattr__(self, attr):
            return getattr(self._optimizer, attr)

    return OptimizerWrapper(optimizer)
