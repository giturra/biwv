from base import BaseVocab
from collections import defaultdict

class Vocab(BaseVocab):
    
    def __init__(self, max_size):
        super().__init__(max_size)
        self.table = defaultdict(int)
    
    def add(self, word):
        if word not in self.table and not self.is_full():
            word_index = self.current_size
            self.table[word] = word_index
            self.current_size += 1
            return word_index
        elif word in self.table:
            return self.table[word]
        return -1
    
    def __getitem__(self, word):
        if word in self.table:
            return self.table[word]
        return -1