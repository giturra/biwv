import abc

class BaseSimulator:

    def __init__(self, stream, method, f, d):

        self.stream = stream
        self.method = method
        self.f = f
        self.d = d

    @abc.abstractmethod
    def train(self):
        ...
    
    @abc.abstractclassmethod
    def train_with_change(self):
        ...