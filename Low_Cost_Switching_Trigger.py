# File: Low_Cost_Switching_Trigger.py
# Author: Vishwa Parekh <vishwaparekh@gmail.com>

from tensorpack import ProxyCallback, Callback


class ExpPeriodicTrigger(ProxyCallback):
    """
    Trigger a callback every k global steps or every k epochs by its :meth:`trigger()` method.

    Most existing callbacks which do something every epoch are implemented
    with :meth:`trigger()` method. By default the :meth:`trigger()` method will be called every epoch.
    This wrapper can make the callback run at a different frequency.

    All other methods (``before/after_run``, ``trigger_step``, etc) of the given callback
    are unaffected. They will still be called as-is.
    """
    def __init__(self, triggerable, every_k_steps=None, exponential_decay = 2, every_k_epochs=None, before_train=False, ):
        """
        Args:
            triggerable (Callback): a Callback instance with a trigger method to be called.
            every_k_steps (int): trigger when ``global_step % k == 0``. Set to
                None to ignore.
            every_k_epochs (int): trigger when ``epoch_num % k == 0``. Set to
                None to ignore.
            before_train (bool): trigger in the :meth:`before_train` method.

        every_k_steps and every_k_epochs can be both set, but cannot be both None unless before_train is True.
        """
        assert isinstance(triggerable, Callback), type(triggerable)
        super(ExpPeriodicTrigger, self).__init__(triggerable)
        if before_train is False:
            assert (every_k_epochs is not None) or (every_k_steps is not None), \
                "Arguments to PeriodicTrigger have disabled the triggerable!"
        self._step_k = every_k_steps
        self._epoch_k = every_k_epochs
        self._do_before_train = before_train
        self._exponential_decay = exponential_decay


    def _before_train(self):
        self.cb.before_train()
        if self._do_before_train:
            self.cb.trigger()

    def _trigger_step(self):
        self.cb.trigger_step()
        if self._step_k is None:
            return
        
        if self.global_step % self._step_k == 0:
            print(self._step_k)
            self._step_k = self._step_k * self._exponential_decay
            self.cb.trigger()

    def _trigger_epoch(self):
        if self._epoch_k is None:
            return
        if self.epoch_num % self._epoch_k == 0:
            self.cb.trigger()

    def __str__(self):
        return "PeriodicTrigger-" + str(self.cb)
