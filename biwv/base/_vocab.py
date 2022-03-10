import abc 


class BaseVocab:


    def __init__(self, max_size):
        self.max_size = max_size
        self.current_size = 0
        # self.table = dict()

    def is_full(self):
        return self.current_size == self.max_size
    
    def is_empty(self):
        return self.current_size == 0
    
    @abc.abstractclassmethod
    def add(self, word):
        ...
    
    @abc.abstractclassmethod
    def remove(self, word):
        ...
    
    @abc.abstractclassmethod
    def __len__(self):
        ...
    
    @abc.abstractmethod
    def __getitem__(self, word):
        ...

    @abc.abstractclassmethod
    def __repr__(self):
        ...
    
    @abc.abstractclassmethod
    def __contains__(self, word):
        ...