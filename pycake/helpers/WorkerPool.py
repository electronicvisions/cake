import multiprocessing
import time
import traceback

workers = multiprocessing.cpu_count() - 1

class WorkerPool(object):
    def __init__(self, f, workers = None):
        if workers is None:
            workers = multiprocessing.cpu_count()

        self.ii = 0
        self.q_in = multiprocessing.JoinableQueue()
        manager = multiprocessing.Manager()
        self.result = manager.list()
        self.workers = [self._make_worker(f) for ii in range(workers)]
        for w in self.workers:
            w.start()

    def __del__(self):
        for w in self.workers:
            w.terminate()

    def do(self, *args):
        self.q_in.put_nowait( (self.ii, args) )
        self.ii += 1

    def join(self):
        for w in self.workers:
            self.q_in.put(None)
        self.q_in.join()
        self.q_in.close()

        result = list(self.result)
        assert len(result) == self.ii
        result.sort()
        result = [r[1] for r in result]
        for r in result:
            if isinstance(r, Exception):
                raise r

        for w in self.workers:
            w.join(0.5)
            if w.is_alive():
                print "Fucking Zombies..."
            w.terminate()
        return result

    def _make_worker(self, f):
        return multiprocessing.Process(
            target=self._process,
            args = (f, self.q_in, self.result)
            )

    @staticmethod
    def _process(f, q_in, out_list):
        for ii, args in iter(q_in.get, None):
            try:
                out_list.append((ii, f(*args)))
            except Exception as e:
                print e
                traceback.print_exc()
                out_list.append( (ii, e) )
            finally:
                q_in.task_done()
        else:
            q_in.task_done()
            q_in.close()
            q_in.join_thread()
