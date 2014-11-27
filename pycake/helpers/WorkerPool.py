import multiprocessing
import time
import traceback
import cPickle

workers = multiprocessing.cpu_count() - 1

class WorkerPool(object):
    """A worker pool"""

    def __init__(self, f, workers = None):
        if workers is None:
            workers = multiprocessing.cpu_count()

        self.q_in = multiprocessing.JoinableQueue()
        manager = multiprocessing.Manager()
        self.result = manager.dict()
        self.workers = [self._make_worker(f) for ii in range(workers)]
        for w in self.workers:
            w.start()

    def __del__(self):
        for w in self.workers:
            w.terminate()

    def __enter__(self):
        return self

    def __exit__(self, error_type, value, traceback):
        self.terminate()

    def is_alive(self):
        return all(w.is_alive() for w in self.workers)

    def do(self, key, *args):
        if not self.is_alive():
            self.terminate()
            raise RuntimeError("Worker process died unexpeted.")
        self.q_in.put_nowait((key, args))

    def join(self):
        for w in self.workers:
            self.q_in.put(None)
        # TODO potential dead lock, if a worker or queue process dies unexpectedly
        if not self.is_alive():
            self.terminate()
            raise RuntimeError("Worker process died unexpeted.")
        self.q_in.join()
        self.q_in.close()

        result = dict(self.result)
        for k, r in result.iteritems():
            if isinstance(r, Exception):
                raise r

        for w in self.workers:
            w.join(0.5)
            if w.is_alive():
                print "Fucking Zombies..."
            w.terminate()
        return result

    def terminate(self):
        for w in self.workers:
            w.terminate()

    def _make_worker(self, f):
        return multiprocessing.Process(
            target=self._process,
            args = (f, self.q_in, self.result)
            )

    @staticmethod
    def _process(f, q_in, out_list):
        for key, args in iter(q_in.get, None):
            try:
                out_list[key] = f(*args)
            except Exception as e:
                import traceback
                print "{}\nERROR in worker thread:\n{} ({})\n{}\n{}".format(
                    '-' * 80, e, type(e), traceback.format_exc(), '-' * 80)
                try:
                    out_list[key] = e
                except cPickle.PicklingError:
                    out_list[key] = RuntimeError(str(e))
            finally:
                q_in.task_done()
        else:
            q_in.task_done()
            q_in.close()
            q_in.join_thread()
