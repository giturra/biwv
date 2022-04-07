import numpy as np
from scipy import sparse
from functools import partial

from base.iwv import IncrementalWordVector
from .vocab import Context, Vocab


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

        # self.vocab.add(WordRep('unk', self.context_size))
    
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
                print(f'{self.vocab.index2word[word]} {self.get_embedding(self.vocab.index2word[word])}')
    
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

    # def get_embedding(self, word):
    #     if word in self.vocab:
    #         word_rep = self.vocab[word]
    #         embedding = np.zeros(self.context_size, dtype=float)
    #         contexts = word_rep.contexts.items()
    #         #print(contexts)
    #         if self.is_ppmi:
    #             for context, coocurence in contexts:
    #                 #print(context, coocurence)
    #                 ind_c = self.contexts[context]
    #                 #print(ind_c)
    #                 #print(f'es un word rep {word_rep}')
    #                 #print(context, coocurence)
    #                 pmi = np.log2(
    #                     (coocurence * self.d) / (word_rep.counter * self.vocab[context].counter) 
    #                 )
    #                 #print(f'ppmi = {pmi}')
    #                 embedding[ind_c] = max(0, pmi)
    #         else:
    #             for context, coocurence in contexts:
    #                 ind_c = self.contexts[context]
    #                 embedding[ind_c] = coocurence 
    #         return embedding
    #     False

# def get_contexts(ind_word, w_size, tokens):
#     # to do: agregar try para check que es posible obtener los elementos de los tokens
#     slice_start = ind_word - w_size if (ind_word - w_size >= 0) else 0
#     slice_end = len(tokens) if (ind_word + w_size + 1 >= len(tokens)) else ind_word + w_size + 1
#     first_part = tokens[slice_start: ind_word]
#     last_part = tokens[ind_word + 1: slice_end]
#     contexts = tuple(first_part + last_part)
#     return contexts

def context_windows(region, left_size, right_size):
    """generate left_context, word, right_context tuples for each region

    Args:
        region (str): a sentence
        left_size (int): left windows size
        right_size (int): right windows size
    """

    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = window(region, start_index, i - 1)
        right_context = window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def window(region, start_index, end_index):
    """Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.

    Args:
        region (str): the sentence for extracting the token base on the context
        start_index (int): index for start step of window
        end_index (int): index for the end step of window
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):
                             min(end_index, last_index) + 1]
    return selected_tokens



def _counts2PPMI(context_index, target_index, total, coor_matrix, counts):
    return max(
        np.log2(
            (coor_matrix[target_index, context_index] * total) / (counts[target_index] * counts[context_index])
        ), 0
    )