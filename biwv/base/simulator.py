import abc

class BaseSimulator:

    def __init__(self, stream, method, f, d, change=False):

        self.stream = stream
        self.method = method
        self.f = f
        self.d = d
        self.with_change = change
        self.counter = 0

    @abc.abstractmethod
    def train(self):
        ...
    
    @abc.abstractclassmethod
    def train_with_change(self):
        ...