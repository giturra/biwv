from typing import Counter
from collections import defaultdict
from base import BaseVocab


class Vocab(BaseVocab):

    def __init__(self, max_size):
        super().__init__(max_size)
        self.table = defaultdict(int)

    def add(self, word):
        if word not in self.table.keys() and not self.is_full():
            self.table[word.word] = word
            self.table[word.word].counter += 1
            self.current_size += 1
        elif word in self.table.keys():
            self.table[word.word].counter += 1

    def __repr__(self):
        return self.table.keys().__repr__()

    def __str__(self):
        return self.table.keys().__str__()            

class Context(BaseVocab):

    def __init__(self, max_size):
        super().__init__(max_size)
        self.table = defaultdict(int)
        self.table['unk'] = 0
        self.current_size += 1


    def add(self, word):
        if word not in self.table and not self.is_full():
            self.table[word] = self.current_size
            self.current_size += 1
    
    def __getitem__(self, word):
        return self.table[word]
    
class WordRep:

    def __init__(self, word, c_size):
        self.word = word
        self.max_size = c_size
        self.size = 0
        self.contexts = defaultdict(int)
        self.counter = 0

        # checking
        self.contexts['unk'] = 0
        self.size += 1

    def is_full(self):
        return self.size == self.max_size   

    def add_context(self, context):
        if context in self.contexts or self.is_full():
            print(context in self.contexts)
            if context in self.contexts.keys():
                print(context, self.contexts[context])
                self.contexts[context] += 1
                print(context, self.contexts[context])
            else:
                self.contexts['unk'] += 1
        # I'm sure if this necesary this condition maybe for hashing
        elif self.size == self.max_size:
            self.contexts['unk'] += 1
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