from __future__ import absolute_import, print_function

import time


class TimeIt:
    def __init__(self, prompt, auto_start=True):
        self.prompt = prompt
        self.auto_start = auto_start
        self.sum = 0
        self.timming = False
    def __enter__(self):
        if self.auto_start:
            self.start()
        return self
    
    def start(self):
        if self.timming:
            raise Exception("[%s] You can't start timing because the timing is not yet stopped." % self.prompt)
        self.timming = True
        self.start_t = time.time()
    
    def stop(self):
        if not self.timming:
            raise Exception("[%s] You can't stop timming because the timing haven't started." % self.prompt)
        self.sum += time.time() - self.start_t
        self.timming = False

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.timming:
            self.stop()
        if exc_tb is None:
            print("[%s] consumes %.6fs" % (self.prompt, self.sum))
        else:
            print("Exception in ", self.prompt)


if __name__ == "__main__":
    with TimeIt("test", auto_start=False) as t:
        t.start()
        print("Hello")
        t.stop()
