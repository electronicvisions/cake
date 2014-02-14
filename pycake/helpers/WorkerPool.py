import multiprocessing
import time

workers = multiprocessing.cpu_count() - 1

class WorkerPool(object):
    def __init__(self, f, workers = None):
        if workers is None:
            workers = multiprocessing.cpu_count()

        self.q_in = multiprocessing.JoinableQueue()
        self.q_out = multiprocessing.JoinableQueue()
        self.ii = 0
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
        self.q_in.join()
        for w in self.workers:
            self.q_in.put(None)
        for w in self.workers:
            w.join(0.5)
            if w.is_alive():
                print "Fucking Zombies..."
            w.terminate()
        result = [None for ii in range(self.ii)]
        for ii in range(self.ii):
            pos, res = self.q_out.get(False)
            if isinstance(res, Exception):
                raise res
            result[pos] = res
        return result

    def _make_worker(self, f):
        return multiprocessing.Process(
            target=self._process,
            args = (f, self.q_in, self.q_out)
            )

    @staticmethod
    def _process(f, q_in, q_out):
        print "START WORKER"
        for ii, args in iter(q_in.get, None):
            try:
                print "worker:", ii, "start"
                q_out.put((ii, f(*args)))
            except Exception as e:
                print e
                q_out.put( (ii, e) )
            finally:
                q_in.task_done()
                print "worker:", ii, "DONE"
        else:
            print "STOP WORKER"


    def _process_x(f, q_in, q_out):
        print "START WORKER"
        while True:
            try:
                x = q_in.get()
                if x is None:
                    print "STOP WORKER"
                    q_in.task_done()
                    return
                ii, args = x
                print "worker:", ii, "start"
                q_out.put((ii, f(*args)))
            except Exception as e:
                print e
                q_out.put( (ii, e) )
            q_in.task_done()
            print "worker:", ii, "DONE"

