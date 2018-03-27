import time

class Timer():
    def __init__(self, active):
        self.time = time.time()
        self.start_time = time.time()
        self.active = active
        
    def timeit(self, msg = None):
        if msg and self.active:
            print('--------------------Time : ', '{0:.2f}'.format(time.time()*1000.0 - self.time*1000.0) , 'ms used for ____', msg )
        self.time = time.time()
        
    def total_time(self, msg = ''):
        if self.active:
            print('--------------------Total time used '+msg+' : ', '{0:.2f}'.format(time.time()*1000.0 - self.start_time*1000.0), ' ms')
