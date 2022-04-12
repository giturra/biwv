import numpy as np
from scipy import sparse
from functools import partial
from river.utils import numpy2dict


from base import IncrementalWordVector
from .vocab import Context, Vocab

from utils import _counts2PPMI, context_windows


class WordContextMatrix(IncrementalWordVector):

    def __init__(
        self, 
        v_size, 
        c_size, 
        w_size, 
        normalize=True,
        on=None,
        strip_accents=True,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        ngram_range=(1, 1),
        is_ppmi=True
    ):
        super().__init__(
            v_size,
            c_size,
            w_size,
            on=on,
            normalize=normalize, 
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
        )

        self.vocab = Vocab(self.vocab_size)
        self.contexts = Context(self.context_size)
        self.coocurence_matrix = sparse.lil_matrix((self.vocab_size, self.context_size))

        self.d = 0

        self.is_ppmi = is_ppmi
    
    def learn_one(self, x, **kwargs):
        tokens = self.process_text(x)
        if len(self.vocab) <= self.vocab_size: 
            for w in tokens:
                self.vocab.add(w)
        if len(self.contexts) <= self.context_size:
            for w in tokens:
                self.contexts.add(w)
        idxs = self.tokens2idxs(tokens)
        for left_context, word, right_context in context_windows(idxs, self.window_size, self.window_size):
            self.d += 1
            if word == -1:   
                continue
            else:
                for i, context_word in enumerate(left_context[::-1]):
                    if context_word in self.contexts:
                        self.coocurence_matrix[word, context_word] += 1.0
                for i, context_word in enumerate(right_context):
                    if context_word in self.contexts:
                        print(context_word)
                        self.coocurence_matrix[word, context_word] += 1.0
            print(f'{self.vocab.index2word[word]} {self.get_embedding(self.vocab.index2word[word])}')

    def learn_many(self, X, y=None, **kwargs):
        for x in X:
            tokens = self.process_text(x)
            if len(self.vocab) <= self.vocab_size: 
                for w in tokens:
                    self.vocab.add(w)
            if len(self.contexts) <= self.context_size:
                for w in tokens:
                    self.contexts.add(w)
            idxs = self.tokens2idxs(tokens)
            for left_context, word, right_context in context_windows(idxs, self.window_size, self.window_size):
                self.d += 1
                if word == -1:   
                    continue
                else:
                    for i, context_word in enumerate(left_context[::-1]):
                        if context_word in self.contexts:
                            self.coocurence_matrix[word, context_word] += 1.0
                    for i, context_word in enumerate(right_context):
                        if context_word in self.contexts:
                            print(context_word)
                            self.coocurence_matrix[word, context_word] += 1.0
                #print(f'{self.vocab.index2word[word]} {self.get_embedding(self.vocab.index2word[word])}')
    
    def tokens2idxs(self, tokens):
        idxs = []
        for token in tokens:
            idxs.append(self.vocab[token])
        return idxs
    
    # todo preguntar al pablo si esto es mala práctica.

 


    def transform_one(self, x: dict):
    
        ...

    def get_embedding(self, word):
        if word in self.vocab:
            idx = self.vocab.word2idx[word]
            c2p = partial(
                _counts2PPMI, target_index=idx, total=self.d, coor_matrix=self.coocurence_matrix, counts=self.vocab.counts 
            )
            contexts_ids = list(self.contexts.index2word.keys())
            return np.array(list(map(c2p, contexts_ids)))

    def embedding2dict(self, word):
        emb = self.get_embedding(word)
        return numpy2dict(emb)
    
    def vocab2dict(self):
        embeddings = {}
        for word in self.vocab.word2idx.keys():
            embeddings[word] = self.get_embedding(word)
        return embeddings





