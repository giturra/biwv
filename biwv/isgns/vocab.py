from base import BaseVocab
from collections import defaultdict

class Vocab(BaseVocab):
    
    def __init__(self, max_size):
        super().__init__(max_size)
        self.table = defaultdict(int)
    
    def add(sel)