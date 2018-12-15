import time

class TimeIt:
    def __init__(self, prompt):
        self.prompt = prompt
    def __enter__(self):
        self.start = time.time()
        #print(self.prompt, "is running")
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.end = time.time()
        if exc_tb is None:
            print("[%s] consumes %.6fs" % (self.prompt, self.end - self.start))
        else:
            print("Exception in ", self.prompt)


if __name__ == "__main__":
    with TimeIt("test"):
        print("Hello")