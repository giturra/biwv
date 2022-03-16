from base._vocab import BaseVocab

class Vocabulary(BaseVocab):

    def __init__(self, max_size):
        super().__init__(max_size)
    
    def add(self, word):
        if word not in self.table and not self.is_full():
            word_index = self.current_size
            self.table[word] = word_index
            self.current_size += 1
            return word_index

        elif word in self.table:
            word_index = self.table[word]
            return word_index
        else:
            return -1
                    
    def __getitem__(self, word: str):
        if word in self.table:
            word_index = self.table[word] 
            return word_index
        return -1