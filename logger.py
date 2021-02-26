import os
import sys


class Logger():
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger("log.txt")

# print("Jack Cui")
# print("https://cuijiahua.com")
# print("https://mp.weixin.qq.com/s/OCWwRVDFNslIuKyiCVUoTA")