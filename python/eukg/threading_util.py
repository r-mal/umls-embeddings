import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor


condition = threading.Condition()
def synchronized(func):
  condition.acquire()
  r = func()
  condition.notify()
  condition.release()
  return r


def parallel_stream(iterable, parallelizable_fn, future_consumer=None):
  """
  Executes parallelizable_fn on each element of iterable in parallel. When each thread finishes, its future is passed
  to future_consumer if provided.
  :param iterable: an iterable of objects to be processed by parallelizable_fn
  :param parallelizable_fn: a function that operates on elements of iterable
  :param future_consumer: optional consumer of objects returned by parallelizable_fn
  :return: void
  """
  if future_consumer is None:
    future_consumer = lambda f: f.result()

  num_threads = multiprocessing.cpu_count()
  executor = ThreadPoolExecutor(max_workers=num_threads)
  futures = []
  for tup in iterable:
    # all cpus are in use, wait until at least 1 thread finishes before spawning new threads
    if len(futures) >= num_threads:
      full_queue = True
      while full_queue:
        incomplete_futures = []
        for fut in futures:
          if fut.done():
            future_consumer(fut)
            full_queue = False
          else:
            incomplete_futures.append(fut)
        futures = incomplete_futures
    # spawn new thread
    futures += [executor.submit(parallelizable_fn, tup)]

  # ensure all threads have finished
  for fut in futures:
    future_consumer(fut)