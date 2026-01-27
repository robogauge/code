import multiprocessing
import multiprocessing.pool

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        """ context=ctx in kwargs is required """
        super(NoDaemonPool, self).__init__(*args, **kwargs)

    def Process(self, *args, **kwds):
        proc = super(NoDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc
