import signal

class TimedOutExc(Exception):
  pass

def deadline(timeout, *args):
  def decorate(f):
    def handler(signum, frame):
      raise TimedOutExc()

    def new_f(*args):

      signal.signal(signal.SIGALRM, handler)
      signal.alarm(timeout)
      return f(*args)
      signa.alarm(0)

    new_f.__name__ = f.__name__
    return new_f
  return decorate
