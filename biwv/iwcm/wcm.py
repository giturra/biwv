import numpy as np
from scipy import sparse
from functools import partial
from river.utils import numpy2dict
from torch import embedding


from base import IncrementalWordVector
from .vocab import Context, Vocab

from utils import _counts2PPMI, context_windows, csr_row_set_nz_to_val


class WordContextMatrix(IncrementalWordVector):

    def __init__(
        self, 
        v_size, 
        c_size, 
        w_size,
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
        for w in tokens:
            self.d += 1
            self.vocab.add(w)
        for w in tokens:
            i = tokens.index(w)
            contexts = _get_contexts(i, self.window_size, tokens)
            if w not in self.vocab:
                w = 'unk'
            for c in contexts:
                self.contexts.add(c)
                row = self.vocab.word2idx.get(w, 0)
                col = self.contexts.word2idx.get(c, 0)
                self.coocurence_matrix[row, col] += 1
    
    def learn_many(self, X, y=None, **kwargs):
        for x in X:
            tokens = self.process_text(x)
            for w in tokens:
                self.d += 1
                self.vocab.add(w)
            for w in tokens:
                i = tokens.index(w)
                contexts = _get_contexts(i, self.window_size, tokens)
                if w not in self.vocab:
                    w = 'unk'
                for c in contexts:
                    self.contexts.add(c)
                    row = self.vocab.word2idx.get(w, 0)
                    col = self.contexts.word2idx.get(c, 0)
                    self.coocurence_matrix[row, col] += 1                
                    
    # def learn_one(self, x, **kwargs):
    #     tokens = self.process_text(x)
    #     if len(self.vocab) <= self.vocab_size: 
    #         for w in tokens:
    #             self.vocab.add(w)
    #     if len(self.contexts) <= self.context_size:
    #         for w in tokens:
    #             self.contexts.add(w)
    #     idxs = self.tokens2idxs(tokens)
    #     for left_context, word, right_context in context_windows(idxs, self.window_size, self.window_size):
    #         self.d += 1
    #         if word == -1:   
    #             continue
    #         else:
    #             for i, context_word in enumerate(left_context[::-1]):
    #                 if context_word in self.contexts:
    #                     self.coocurence_matrix[word, context_word] += 1.0
    #             for i, context_word in enumerate(right_context):
    #                 if context_word in self.contexts:
    #                     print(context_word)
    #                     self.coocurence_matrix[word, context_word] += 1.0
    #         print(f'{self.vocab.index2word[word]} {self.get_embedding(self.vocab.index2word[word])}')

    # def learn_many(self, X, y=None, **kwargs):
    #     for x in X:
    #         tokens = self.process_text(x)
    #         if len(self.vocab) <= self.vocab_size: 
    #             for w in tokens:
    #                 self.vocab.add(w)
    #         if len(self.contexts) <= self.context_size:
    #             for w in tokens:
    #                 self.contexts.add(w)
    #         idxs = self.tokens2idxs(tokens)
    #         for left_context, word, right_context in context_windows(idxs, self.window_size, self.window_size):
    #             self.d += 1
    #             if word == -1:   
    #                 continue
    #             else:
    #                 for i, context_word in enumerate(left_context[::-1]):
    #                     if context_word in self.contexts:
    #                         self.coocurence_matrix[word, context_word] += 1.0
    #                 for i, context_word in enumerate(right_context):
    #                     if context_word in self.contexts:
    #                         # print(context_word)
    #                         self.coocurence_matrix[word, context_word] += 1.0
    
    def tokens2idxs(self, tokens):
        idxs = []
        for token in tokens:
            idxs.append(self.vocab.word2idx.get(token, -1))
        return idxs
    
    # todo preguntar al pablo si esto es mala práctica.

 


    def transform_one(self, x: dict):
    
        ...

    def get_embedding(self, word):
        if word in self.vocab:
            vidx = self.vocab.word2idx.get(word, 0)
            contexts_ids = self.coocurence_matrix[vidx].nonzero()[1]
            embedding = np.zeros(self.context_size, dtype=float)
            for cidx in contexts_ids:
                value = np.log2(
                    (self.d * self.coocurence_matrix[vidx, cidx]) / (self.vocab.counter[vidx] * self.contexts.counter[cidx])
                )
                embedding[cidx] = max(0.0, value)
            return embedding

    # def get_embedding(self, word):
    #     if word in self.vocab:
    #         idx = self.vocab.word2idx.get(word, 0)
    #         c2p = partial(
    #             _counts2PPMI, target_index=idx, total=self.d, coor_matrix=self.coocurence_matrix, 
    #             v_counts=self.vocab.counter, c_counts=self.contexts.counter 
    #         )
    #         contexts_ids = list(self.contexts.index2word.keys())
    #         return np.array(list(map(c2p, contexts_ids)))

    def embedding2dict(self, word):
        if word in self.vocab:
            emb = self.get_embedding(word)
            output = numpy2dict(emb) 
            return output
        raise ValueError(f"{word} not found in vocabulary.")
    
    def vocab2dict(self):
        embeddings = {}
        for word in self.vocab.word2idx.keys():
            embeddings[word] = self.get_embedding(word)
        return embeddings





def _get_contexts(ind_word, w_size, tokens):
    # to do: agregar try para check que es posible obtener los elementos de los tokens
    slice_start = ind_word - w_size if (ind_word - w_size >= 0) else 0
    slice_end = len(tokens) if (ind_word + w_size + 1 >= len(tokens)) else ind_word + w_size + 1
    first_part = tokens[slice_start: ind_word]
    last_part = tokens[ind_word + 1: slice_end]
    contexts = tuple(first_part + last_part)
    return contexts