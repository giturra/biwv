from typing import Counter
from base import BaseVocab
from dsa import SpaceSavingAlgorithm

class Vocab(BaseVocab):

    def __init__(self, max_size, ssa=False):
        super().__init__(max_size)
        self.index2word = {}
        self.word2idx = {}
        self.ssa = ssa
        if not ssa:
            self.counts = Counter()
        else:
            self.sscounter = SpaceSavingAlgorithm(max_size)

        
    def add(self, word):
        if not self.ssa:
            self._add(word)
        else:
            self._add_dsa(word)
    
    def _add_dsa(self, word):
        ...

    def _add(self, word):
        if word not in self.word2idx.keys() and not self.is_full():
            self.word2idx[word] = self.current_size
            self.index2word[self.current_size] = word
            self.counts[self.word2idx[word]] = 1
            self.current_size += 1
        elif word in self.word2idx.keys():
            self.counts[self.word2idx[word]] += 1

    def __repr__(self):
        return self.word2idx.keys().__repr__()

    def __str__(self):
        return self.word2idx.keys().__str__()

    def __getitem__(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        return -1      

    def __len__(self):
        return len(self.word2idx.keys())

    def __contains__(self, word):
        return word in self.word2idx.keys()      

class Context(BaseVocab):

    def __init__(self, max_size):
        super().__init__(max_size)
        self.index2word = {}
        self.word2idx = {}

    def add(self, word):
        if word not in self.word2idx and not self.is_full():
            self.word2idx[word] = self.current_size
            self.index2word[self.current_size] = word
            self.current_size += 1
    
    # def __getitem__(self, word):
    #     return self.word2idx[word]
    
    def __repr__(self):
        return self.word2idx.keys().__repr__()

    def __str__(self):
        return self.word2idx.keys().__str__()

    def __len__(self):
        return len(self.word2idx.keys())

    def __contains__(self, word):
        return word in self.index2word.keys() 
    
# class WordRep:

#     def __init__(self, word, c_size):
#         self.word = word
#         self.max_size = c_size
#         self.size = 0
#         self.contexts = defaultdict(int)
#         self.counter = 0

#         # checking
#         self.contexts['unk'] = 0
#         self.size += 1

#     def is_full(self):
#         return self.size == self.max_size   

#     def add_context(self, context):
#         if context in self.contexts or self.is_full():
#             #print(context in self.contexts)
#             if context in self.contexts.keys():
#                 #print(context, self.contexts[context])
#                 self.contexts[context] += 1
#                 #print(context, self.contexts[context])
#             else:
#                 self.contexts['unk'] += 1
#         # I'm not sure if this is necesary this condition maybe for hashing
#         elif self.size == self.max_size:
#             self.contexts['unk'] += 1
#         else:
#             self.contexts[context] += 1
#             self.size += 1
    
#     def __len__(self):
#         return len(self.contexts.keys())

#     def __repr__(self):
#         return self.word
    
#     def __getitem__(self, context):
#         if context in self.contexts:
#             return self.contexts[context]
#         return self.contexts['unk'] 