import sys
import threading
import os

class myThread (threading.Thread):
    def __init__(self, file):
        threading.Thread.__init__(self)
        self.file = file
    def run(self):
        os.system("python3 " + self.file)

d = {}

for i in range(1,len(sys.argv)):
    d[i] = myThread(sys.argv[i])
    
for i in range(1,len(sys.argv)):
    d[i].start()

for i in range(1,len(sys.argv)):
    d[i].join()

