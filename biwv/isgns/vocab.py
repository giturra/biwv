from base import BaseVocab

class Vocab(BaseVocab):

    def __init__(self, max_size):
        super().__init__(max_size)
        self.word2idx = dict()
    
    def add(self, word):
        if word not in self.word2idx and not self.is_full():
            word_index = self.current_size
            self.word2idx[word] = word_index
            self.current_size += 1
            return word_index

        elif word in self.word2idx:
            word_index = self.word2idx[word]
            return word_index
        else:
            return -1
                    
    def __getitem__(self, word: str):
        if word in self.word2idx:
            word_index = self.word2idx[word] 
            return word_index
        return -1
    
    def __contains__(self, word):
        return word in self.word2idx