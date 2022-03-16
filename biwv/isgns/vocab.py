from typing import Counter
from base import BaseVocab


class Vocab(BaseVocab):
    
    def __init__(self, max_size):
        super().__init__(max_size)
        self.table = {}
        self.counter = Counter()
        self.total_counts = 0
    
    def add(self, word):
        if word not in self.table and not self.is_full():
            word_index = self.current_size
            self.table[word] = word_index
            self.counter.update([word])
            self.current_size += 1
            self.total_counts += 1
            return word_index
        elif word in self.table:
            self.counter.update([word])
            self.total_counts += 1
            return self.table[word]
        return -1

    def __getitem__(self, word):
        if word in self.table:
            return self.table[word]
        return -1