import multiprocessing
import time
import traceback
import cPickle
import pylogging
import os

logger = pylogging.get("pycake.helper.workerpool")

CPUS = os.environ.get('SLURM_JOB_CPUS_PER_NODE', multiprocessing.cpu_count())
TASKS = os.environ.get('SLURM_TASKS_PER_NODE', 1)
DEFAULT_WORKERS = -(-int(CPUS) // int(TASKS)) # Round upwards

def process(f, key, *args, **kwargs):
    """
    Helper function to be called by the process pool. This function will
    print any exception and then reraise it. This is helpful in debugging,
    because AsyncResult.get will not print the stacktrace of the subprocess.
    """
    try:
        return key, f(*args, **kwargs)
    except Exception as e:
        import traceback
        print "{}\nERROR in worker thread:\n{} ({})\n{}\n{}".format(
            '-' * 80, e, type(e), traceback.format_exc(), '-' * 80)
        # Exception is propagated by worker tree
        raise

class WorkerPool(object):
    """A worker pool

    If a child process of multiprocessing.Pool dies, e.g. due to memory
    restrictions, the join method will wait forever. This WorkerPool is designed
    to avoid this. We keep track of the pids of the worker and can check
    if any died.
    """

    def __init__(self, f, workers=DEFAULT_WORKERS):
        logger.info("Initialize worker pool with {} workers".format(workers))

        self.pool = multiprocessing.Pool(workers)
        self.poolpids = tuple(proc.pid for proc in self.pool._pool)
        self.callback = f
        self.tasks = []
        self.results = []

    def __del__(self):
        self.pool.terminate()

    def __enter__(self):
        return self

    def __exit__(self, error_type, value, traceback):
        self.terminate()

    def currentpids(self):
        """Returns process pids of the workers currently used by the pool"""
        return tuple(proc.pid for proc in self.pool._pool)

    def is_alive(self):
        # If a worker dies a new one is started, but the thread pool is not
        # working reliable anymore.
        return self.poolpids == self.currentpids()

    def do(self, key, *args, **kwargs):
        if not self.is_alive():
            pids = list(set(self.poolpids) - set(self.currentpids()))
            raise SystemError("Worker processes {} died unexpectedly.".format(
                pids))

        logger.debug("Add task {}".format(key))
        self.tasks.append(self.pool.apply_async(
            process, (self.callback, key) + args, kwargs))

        # Start empting the queue, this allows to free memory needed by
        # traces. Also we will see exceptions in workers earlier
        while self.tasks and self.tasks[0].ready():
            self.results.append(self.tasks.pop(0).get())

    def join(self):
        self.pool.close()
        # We want to raise exceptions from subprocesses early, so we start
        # by checking the first element.
        # It is important to use get only when the result is ready, otherwise
        # the child process might have died and we are stuck in the get method
        # forever
        while self.tasks:
            if self.tasks[0].ready():
                self.results.append(self.tasks.pop(0).get())
            elif not self.is_alive():
                raise SystemError("Worker process died unexpectedly.")
            else:
                time.sleep(0.5)
        self.terminate()
        return dict(self.results)

    def terminate(self):
        self.pool.terminate()
