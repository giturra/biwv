from typing import Counter
from collections import defaultdict
from base import BaseVocab


class Vocab(BaseVocab):

    def __init__(self, max_size):
        super().__init__(max_size)
        self.counts = Counter()
    
    
class WordRep:

    def __init__(self, word, c_size):
        self.word = word
        self.max_size = c_size
        self.size = 0
        self.contexts = defaultdict(int)
        self.count = 0

        # checking
        self.contexts['unk'] = 0
        self.size += 1

    def is_full(self):
        return self.size == self.max_size

    def add_context(self, context):
        if context in self.contexts or self.is_full():
            if context in self.contexts:
                self.contexts[context] += 1
            else:
                self.contexts['unk'] += 1
        elif self.size + 1 == self.max_size:
            self.contexts['unk'] += 1
            self.size += 1
        else:
            self.contexts[context] += 1
            self.size += 1
    
    def __len__(self):
        return len(self.contexts.keys())

    def __repr__(self):
        return self.word
    
    def __getitem__(self, context):
        if context in self.contexts:
            return self.contexts[context]
        return self.contexts['unk'] 